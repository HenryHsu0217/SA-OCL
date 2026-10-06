# SA-OCL

Implementation of **Scene-Agnostic Object-Centric Representation Learning for 3D Gaussian Splatting**, by Tsuheng Hsu, Guiyu Liu, Juho Kannala, and Janne Heikkilä. Reference: [arXiv:2604.09045v1](https://arxiv.org/abs/2604.09045v1). The project combines a dataset-wide GOLD object codebook with per-scene 3D Gaussian optimization and object segmentation.

This guide covers environment installation, CUDA builds, dataset preparation, GOLD training, Gaussian training/rendering/evaluation, and result aggregation. Commands use paths relative to the project root; replace `/path/to/SA-OCL` with your checkout location.

`original` identifies inherited Gaussian Grouping code; `paper` contains our implementations and adaptations for the paper; `baselines` contains comparison implementations.

## Contents

- [How the project works](#how-the-project-works)
- [Directory and entrypoint map](#directory-and-entrypoint-map)
- [Environment setup](#environment-setup)
- [Build the CUDA rasterizers](#build-the-cuda-rasterizers)
- [Models and checkpoints](#models-and-checkpoints)
- [Data setup](#data-setup)
- [Preprocessing](#preprocessing)
- [Train GOLD](#train-gold)
- [Run the combined training and evaluation jobs](#run-the-combined-training-and-evaluation-jobs)
- [Run one scene manually](#run-one-scene-manually)
- [Outputs and metrics](#outputs-and-metrics)
- [Baselines and other components](#baselines-and-other-components)
- [Experiment configuration](#experiment-configuration)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## How the project works

There are two training stages. **GOLD learns across scenes**, then its checkpoint is frozen while **each Gaussian scene is optimized separately**. The combined metric launchers perform the second stage, rendering, and evaluation; GOLD training happens beforehand.


1. **Learn global object identities.** [GOLD_Code/train_ocl.py](GOLD_Code/train_ocl.py) trains `GOLD_Code.ocl_origin.OCL`. Slots select learnable object prototypes and combine intrinsic identity features with extrinsic features. Frozen DINO ViT-B/8 provides feature-reconstruction targets. With `--data_name octa`, RGB CNN features supply the attention input. Training includes reconstruction, variational KL terms, and background regularization.
2. **Optimize a Gaussian scene.** The paper trainers freeze GOLD, derive masks and selected codes for each training view, and build a dense identity-feature target. Each Gaussian stores geometry, opacity, RGB spherical harmonics, and a 122-dimensional identity vector. RGB L1/SSIM and feature MSE supervise the scene. `--regloss3d` adds neighborhood identity consistency through KL divergence; densification and pruning remain part of optimization.
3. **Render and identify objects.** The renderer alpha-blends identity vectors with the same visibility used for RGB. Each pixel is assigned to the nearest global prototype by squared L2 distance, with a zero-vector background entry. Slot order therefore does not define output identity.
4. **Evaluate saved views.** The evaluator compares rendered ID masks and RGB images with ground truth, writes per-scene scores, and optionally retains the complete scene output.

The standard paper configuration has 8 slots including background, 11 foreground prototypes, a 128-dimensional foreground slot split into **122 intrinsic + 6 extrinsic** dimensions, and an 8-dimensional background representation. Prototype numbers belong to one checkpoint's vocabulary; independently trained OCTA and GSO codebooks do not automatically share numbering.

| Implementation | Python scene and renderer | Native rasterizer | Identity width |
| --- | --- | --- | --- |
| Our paper method | `core.paper.scene`, `core.paper.gaussian_renderer` | `diff_gaussian_rasterization` | **122** |
| Original Gaussian Grouping and baselines | `core.original.scene_16`, `core.original.gaussian_renderer_16` | `diff_gaussian_rasterization_16` | **16** |

The width is fixed in the Gaussian model and CUDA `NUM_OBJECTS` header. `--num_cls 11` controls GOLD prototypes; the original `--num_classes 256` controls a baseline classifier. Neither changes rasterizer width.

## Directory and entrypoint map

```text
SA-OCL/
├── core/
│   ├── original/                    arguments/utils/ext, scene_16, gaussian_renderer_16
│   └── paper/                       122-feature scene and renderer
├── training/{original,paper,baselines}/
├── rendering/{original,paper,baselines}/
├── preprocessing/
│   ├── original/                    raw-photo COLMAP conversion
│   └── paper/                       known-pose triangulation, SAM/SAM2 preparation
│       └── SAM2/                    tracking script, YAML, weights/
├── evaluation/
│   ├── original/                    inherited RGB/LERF evaluation
│   ├── paper/                       object grouping and RGB metrics
│   ├── notebooks/                   ARI analysis
│   └── reports/                     scores and aggregation notebook
├── editing/original/                removal and inpainting
├── GOLD_Code/                       models, trainers, tests, loaders
│   ├── dino/                        DINO configuration and weights
│   └── vqvae/                       auxiliary reconstruction module
├── third_party/
│   ├── deva/                        DEVA source; weights in saves/
│   └── cuda/                        122/16 rasterizers and simple-knn
├── scripts/
│   ├── env.sh                       project and runtime-library paths
│   ├── run.sh                       Python/Bash entrypoint wrapper
│   ├── original/                    inherited launchers
│   └── paper/{gold,preprocessing,evaluation}/
├── config/gaussian_dataset/train.json
├── data/                            prepared OCTA/GSO datasets
└── logs/, output/, cache/torch/      runtime destinations
```

| Task | Entrypoint |
| --- | --- |
| Train the paper GOLD backbone | [GOLD_Code/train_ocl.py](GOLD_Code/train_ocl.py) |
| Optimize an OCTA Gaussian scene | [training/paper/train_ours.py](training/paper/train_ours.py) |
| Optimize a GSO Gaussian scene | [training/paper/train_GSO.py](training/paper/train_GSO.py) |
| Render paper OCTA / GSO scenes | [render_ours.py](rendering/paper/render_ours.py), [render_ours_GSO.py](rendering/paper/render_ours_GSO.py) |
| Evaluate grouping, segmentation, RGB, FPS | [evaluation/paper/metric.py](evaluation/paper/metric.py) |
| Train Gaussian Grouping / Unified Lift | [train.py](training/original/train.py), [train_unified_lift.py](training/baselines/train_unified_lift.py) |
| Prepare points from known cameras | [preprocessing/paper/to_colmap.py](preprocessing/paper/to_colmap.py) |
| Generate independent SAM / SAM2 masks | [SAM_mask_for_unilift.py](preprocessing/paper/SAM_mask_for_unilift.py), [SAM2_mask_for_unilift.py](preprocessing/paper/SAM2_mask_for_unilift.py) |
| Generate SAM2 tracked masks | [preprocessing/paper/SAM2/sam2_mask.py](preprocessing/paper/SAM2/sam2_mask.py) |

Shared utilities live under `core/original`; using them does not make the paper model a 16-feature model. The active GOLD class is `ocl_origin.OCL`. The similarly named `ocl.OCL` is a separate implementation used by auxiliary trainers/tests.

## Environment setup

### Main training environment

Use Linux, an NVIDIA GPU, a compatible driver, and a CUDA toolkit for native compilation. The configuration below uses Python 3.8 and PyTorch 2.1 with CUDA 12.1. PyTorch's [installation guide for previous versions](https://pytorch.org/get-started/previous-versions/) documents this wheel combination.

Commands use **Bash**. If you use another shell, start `bash` before sourcing `scripts/env.sh`, and initialize Conda for that shell.

```bash
conda create -n gaussian_grouping python=3.8.20 pip=24.2 -y
conda activate gaussian_grouping

python -m pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
    setuptools==75.1.0 wheel ninja==1.13.0 \
    numpy==1.24.3 scipy==1.10.0 scikit-learn==1.3.2 \
    Pillow==10.4.0 opencv-python==4.12.0.88 \
    transformers==4.46.3 huggingface-hub==0.36.0 \
    torchmetrics==1.5.2 lpips==0.1.4 \
    h5py==3.11.0 matplotlib==3.7.5 tqdm==4.67.1 \
    PyYAML==6.0.3 plyfile==0.8.1 einops==0.8.1 \
    pycolmap==3.12.5 tensorboard==2.14.0 wandb==0.23.0

cd /path/to/SA-OCL
source scripts/env.sh
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Activate the environment before sourcing `env.sh`. The helper exposes the project, DEVA, and rasterizer packages on `PYTHONPATH`, sets `TORCH_HOME=cache/torch`, selects the active Conda environment's runtime libraries, and changes to the project root.

Build the native components in the next section, then inspect entrypoint options:

```bash
bash scripts/run.sh GOLD_Code/train_ocl.py --help
bash scripts/run.sh training/paper/train_ours.py --help
bash scripts/run.sh rendering/paper/render_ours.py --help
```

`run.sh` forwards arguments and runs from the project root. `GAUSSIAN_GROUPING_PYTHON` selects its Python interpreter. GOLD and Gaussian training use CUDA. Gaussian `wandb` logging is optional through `--use_wandb`.

### SAM2 and optional preprocessing dependencies

[SAM2's official installation instructions](https://github.com/facebookresearch/sam2) require Python 3.10 or newer and a newer PyTorch stack than the main environment. Keep it separate to preserve the main rasterizers' PyTorch dependency. One CUDA 12.1 setup is:

```bash
conda create -n sam2 python=3.10 pip -y
conda activate sam2
python -m pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/sam2.git'
python -m pip install opencv-python Pillow tqdm
```

The installed package must provide `configs/sam2.1/sam2.1_hiera_b+.yaml`. The YAML in `preprocessing/paper/SAM2/` describes that model; launchers pass the installed package's Hydra configuration name. Return to the main environment before Gaussian training/rendering. SAM2 tracking uses CUDA and BF16 autocasting; select a GPU supporting the required operations.

For optional SAM1 preprocessing, install [Segment Anything](https://github.com/facebookresearch/segment-anything) in its execution environment:

```bash
python -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

DEVA is exposed as vendored source by `env.sh`. Install its dependency list from [third_party/deva/pyproject.toml](third_party/deva/pyproject.toml), without packaging the vendored directory:

```bash
python -m pip install tomli
python - <<'PY'
import subprocess
import sys
import tomli

with open('third_party/deva/pyproject.toml', 'rb') as stream:
    dependencies = tomli.load(stream)['project']['dependencies']
subprocess.check_call([sys.executable, '-m', 'pip', 'install', *dependencies])
PY
```

GroundingDINO-backed DEVA/LERF operations additionally need the [GroundingDINO package and native build](https://github.com/IDEA-Research/GroundingDINO). Install the package as well as placing its checkpoint at the path used by the launcher. The original RGB evaluator imports `lpipsPyTorch`; the paper evaluator uses the separate `lpips` package. Original LaMa inpainting needs its external code/checkpoint paths configured in the editing launcher. These optional components are not required for the GOLD-supervised Gaussian pipeline.

## Build the CUDA rasterizers

The paper method needs the **122-feature rasterizer and `simple-knn`**. Also build the 16-feature rasterizer for Gaussian Grouping or Unified Lift.

| Source directory | Imported native module | Used by |
| --- | --- | --- |
| [third_party/cuda/diff_gaussian_rasterization](third_party/cuda/diff_gaussian_rasterization/) | `diff_gaussian_rasterization._C` | Paper, 122 features |
| [third_party/cuda/diff_gaussian_rasterization_16](third_party/cuda/diff_gaussian_rasterization_16/) | `diff_gaussian_rasterization_16._C` | Original/baselines, 16 features |
| [third_party/cuda/simple-knn](third_party/cuda/simple-knn/) | `simple_knn._C` | Initial Gaussian scales from point distances |

Activate the main environment and make a **CUDA 12.1 toolkit with `nvcc`** and a compatible C++ compiler visible. On a cluster, load matching CUDA/compiler modules. PyTorch wheels supply runtime libraries; compilation additionally needs the toolkit. See PyTorch's [CUDA extension implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.0/torch/utils/cpp_extension.py) for discovery/version checks.

Build on a GPU with the architecture you will use for training, or set `TORCH_CUDA_ARCH_LIST` to the target GPU's compute capability before compilation. To identify the visible GPU's capability:

```bash
python -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}.{minor}")'
```

```bash
conda activate gaussian_grouping
cd /path/to/SA-OCL
source scripts/env.sh
nvcc --version
gcc --version

# Use the actual toolkit prefix if nvcc is provided through a wrapper.
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

# Paper rasterizer: replace the binary inside the source package.
(
    cd third_party/cuda/diff_gaussian_rasterization
    python setup.py clean --all
    python setup.py build_ext --inplace --force
)

python -m pip install --no-build-isolation --no-deps ./third_party/cuda/simple-knn

# Also build this for 16-feature baselines.
(
    cd third_party/cuda/diff_gaussian_rasterization_16
    python setup.py clean --all
    python setup.py build_ext --inplace --force
)
```

**Why build in place:** `env.sh` puts rasterizer source directories before site-packages. Installing a rebuilt wheel alone can leave the bundled source `_C` binary being imported. An in-place build replaces the binary the launchers load. Build using the same Python, PyTorch, CUDA, and GPU architecture as your training environment; rebuild when any of these changes.

Check the paper components:

```bash
python - <<'PY'
import torch
import diff_gaussian_rasterization._C as rasterizer
import simple_knn._C as knn
from core.paper.scene import GaussianModel

print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Rasterizer:', rasterizer.__file__)
print('simple-knn:', knn.__file__)
assert GaussianModel(3).num_objects == 122
PY
```

For baselines, also check `import diff_gaussian_rasterization_16._C` and `core.original.scene_16.GaussianModel(3).num_objects == 16`. Import checks verify dependency loading, not a complete training run.

## Models and checkpoints

Place foundation-model files with the components that load them:

| Component | Model location/files |
| --- | --- |
| DINO ViT-B/8 for GOLD | `GOLD_Code/dino/`: `config.json`, `preprocessor_config.json`, and weights in `model.safetensors` or `pytorch_model.bin` |
| DEVA | `third_party/deva/saves/DEVA-propagation.pth` |
| SAM1 / MobileSAM | `third_party/deva/saves/sam_vit_h_4b8939.pth`, `mobile_sam.pt` |
| GroundingDINO | `third_party/deva/saves/groundingdino_swint_ogc.pth`, `GroundingDINO_SwinT_OGC.py` |
| SAM2.1 base-plus | `preprocessing/paper/SAM2/weights/sam2.1_hiera_base_plus.pt` |

Train GOLD separately for each dataset. The training recipes below save:

| Dataset | GOLD state dictionary |
| --- | --- |
| OCTA | `logs/octa/octa_gold/best_model.pt` |
| GSO | `logs/octa/gso_gold/best_model.pt` |

Set `--checkpoint_path` in both the Gaussian training and rendering commands to the corresponding `best_model.pt`. These are plain model state dictionaries. GOLD's `checkpoint.pt.tar` contains model, optimizer, epoch, and validation state for resuming GOLD. Gaussian optimization uses a separate `chkpnt<N>.pth` format, selected with `--start_checkpoint`.

Keep architecture settings identical between GOLD training, Gaussian training, and rendering. Preserve Gaussian `cfg_args` with its point-cloud directory; rendering reloads saved scene settings from it.

## Data setup

Gaussian `-s` points to the **dataset root**, e.g. `data/GSO_new`, not to `image/`. GOLD `--data_path` points to the scene-organized image root. Numeric scene directory names and frame filenames are part of the loader contract.

### OCTA / OCTScenes_grouped

```text
data/OCTScenes_grouped/
├── poses.tar
└── 128x128/
    ├── intrinsics.tar
    ├── image/
    │   ├── 0000/{00.png,01.png,...,59.png}
    │   ├── ... training scenes through 3099
    │   └── 3100/... through 3199/         validation/evaluation scenes
    ├── GT_mask/
    │   └── 3100-3199/{3100_00.png,...,3199_59.png}
    ├── points/{3100_points3D.ply,...,3199_points3D.ply}
    ├── DEVA_mask/<scene>/<frame>.png       optional baseline masks
    ├── SAM_mask_Uni/<scene>/<frame>.png
    ├── SAM2_mask_Uni/<scene>/<frame>.png
    └── SAM2_track_mask/<scene>/<frame>.png
```

Prepare images for **0–3099 training scenes and 3100–3199 validation/evaluation scenes**, and configure the GOLD split as described in [Train GOLD](#train-gold). Download the source data from [OCTScenes on Hugging Face](https://huggingface.co/datasets/Yinxuan/OCTScenes); the dataset is described in [OCTScenes, arXiv:2306.09682](https://arxiv.org/abs/2306.09682). Arrange the selected scenes and camera inputs in the layout above.

Camera archives are opened directly by the OCTA reader:

- `128x128/intrinsics.tar` members must be `bbox/<scene:04d>/<scene:04d>_<frame>.txt`, holding **3 × 3 intrinsic matrices**. The reader expects the `bbox/` member prefix.
- `poses.tar` members must be `pose/<scene:04d>_<frame>.txt`, holding camera transforms in the convention expected by the reader.
- Images use two-digit frames, e.g. `3100/00.png`; GT uses `3100_00.png` inside the fixed `3100-3199` folder.
- Initial points use `<scene>_points3D.ply`; the loader can fall back to same-prefix COLMAP `.bin` or `.txt` files.

Use calibrated intrinsic matrices for `intrinsics.tar`; bounding-box annotations are a separate input. Keep camera archives in archive form for the readers and inspect their member names with `tar -tf`.

### GSO

Arrange generated GSO data under `data/GSO_new`, or set dataset arguments to the root you use:

```text
data/GSO_new/
├── intrinsic.txt                      shared 3 × 3 intrinsic matrix
├── poses.tar                          pose/<scene:04d>_<frame>.txt
└── 128x128/
    ├── image/<scene:04d>/{00.png,...,59.png}
    ├── GT_mask/{1900_00.png,...,1999_59.png}
    ├── points/{1900_points3D.ply,...,1999_points3D.ply}
    ├── DEVA_mask/<scene>/<frame>.png
    ├── SAM_mask_Uni/<scene>/<frame>.png
    ├── SAM2_mask_Uni/<scene>/<frame>.png
    └── SAM2_track_mask/<scene>/<frame>.png
```

Prepare GSO images for **0–1899 training** and **1900–1999 validation/evaluation**, plus validation GT. GSO GT is flat under `GT_mask/`, unlike OCTA. The GSO reader applies its own camera-coordinate conversion; export poses in the convention expected by that reader.

Generate synthetic scenes with [Kubric](https://github.com/google-research/kubric), following its installation and rendering instructions. Export RGB images, integer instance masks, camera calibration, and camera poses into the layout above.

### Ground truth, pseudo-masks, and splits

GT masks are integer-label grayscale/indexed images: **0 is background**. Colored visualizations cannot replace them. The paper reader loads GT even during Gaussian optimization, although GOLD provides feature supervision; GT is needed for camera records, exported comparison masks, and scoring.

Original/baseline readers select pseudo-masks as follows:

| Flags | Folder under `<size>x<size>/` |
| --- | --- |
| Default Gaussian Grouping | `DEVA_mask` |
| `--uniflift` | `SAM_mask_Uni` |
| `--uniflift --sam2` | `SAM2_mask_Uni` |
| `--sam2` | `SAM2_track_mask` |

`--uniflift` is the actual spelling. Paper commands use `--ours` and GOLD; they do not require these pseudo-mask folders. Use the original 16-feature entrypoints for these baselines.

`--eval --train_split` with default `--split_factor 0.8` chooses 48 training and 12 held-out views from a 60-view scene. This **within-scene split** differs from GOLD's **across-scene split**. Entrypoints seed the selection deterministically; preserve `cfg_args`, image names, and data when rendering later.

Set dataset paths consistently across preprocessing, GOLD, Gaussian training, and rendering. Resize RGB and label masks to the selected resolution while preserving integer labels; camera calibration must correspond to that image resolution.

## Preprocessing

Use existing compatible point clouds directly. To triangulate from known cameras in the main environment:

```bash
bash scripts/run.sh preprocessing/paper/to_colmap.py \
    --type OCT --source_path data/OCTScenes_grouped --size 128

# Alternative dataset:
bash scripts/run.sh preprocessing/paper/to_colmap.py \
    --type GSO --source_path data/GSO_new --size 128
```

The helper processes OCTA 3100–3199 or GSO 1900–1999. Its `random_init=True` branch performs known-pose COLMAP triangulation, writing `colmap_triangulated/<scene>/` and `points/<scene>_points3D.ply`. It requires images, cameras, `pycolmap`, and masks expected by the original reader. [preprocessing/original/convert.py](preprocessing/original/convert.py) handles the separate original raw-photo/COLMAP workflow.

Optional SAM1 masks, with SAM1 installed:

```bash
bash scripts/run.sh preprocessing/paper/SAM_mask_for_unilift.py \
    --img_root data/GSO_new/128x128/image \
    --out_root data/GSO_new/128x128/SAM_mask_Uni \
    --scene_start 1900 --scene_end 1999 \
    --sam_checkpoint third_party/deva/saves/sam_vit_h_4b8939.pth \
    --model_type vit_h
```

Independent SAM2 masks: activate the SAM2 environment and source `scripts/env.sh` again, then run:

```bash
bash scripts/run.sh preprocessing/paper/SAM2_mask_for_unilift.py \
    --img_root data/GSO_new/128x128/image \
    --out_root data/GSO_new/128x128/SAM2_mask_Uni \
    --scene_start 1900 --scene_end 1999 \
    --sam_checkpoint preprocessing/paper/SAM2/weights/sam2.1_hiera_base_plus.pt \
    --sam_config configs/sam2.1/sam2.1_hiera_b+.yaml
```

SAM2 tracked masks of one scene:

```bash
bash scripts/run.sh preprocessing/paper/SAM2/sam2_mask.py \
    --img_dir data/GSO_new/128x128/image \
    --out_dir data/GSO_new/128x128/SAM2_track_mask \
    --scene_id 1900 --reprompt_every 15 \
    --ckpt preprocessing/paper/SAM2/weights/sam2.1_hiera_base_plus.pt \
    --cfg configs/sam2.1/sam2.1_hiera_b+.yaml
```

The project directory `SAM2/` is distinct from the installed lowercase `sam2` package. [scripts/paper/preprocessing](scripts/paper/preprocessing/) contains SLURM versions.

For DEVA masks, configure [GSO_pesudo_label.sh](scripts/paper/preprocessing/GSO_pesudo_label.sh) or [OCTScence_pesudo_label.sh](scripts/paper/preprocessing/OCTScence_pesudo_label.sh): set `OCT_ROOT` to the chosen dataset root, archive paths to your packaged inputs, and `OUT_ROOT` to `<dataset_root>/128x128/DEVA_mask`. Save each scene's masks under `<scene>/<frame>.png` so the baseline reader finds them. Prepare GT labels and camera inputs separately.

## Train GOLD

GOLD follows [Learning global object-centric representations via disentangled slot attention](https://link.springer.com/article/10.1007/s10994-024-06687-9), whose [official implementation](https://github.com/Tonglin-Chen/GOLD) provides the base object-centric model. The paper Gaussian pipeline uses `GOLD_Code.ocl_origin.OCL` through [GOLD_Code/train_ocl.py](GOLD_Code/train_ocl.py).

### Configure the dataset split

In the `if args.data_name == 'octa':` dataset-loading block of `GOLD_Code/train_ocl.py`, set the two `OCTScenesDataset` constructors for the dataset you are training. Scene bounds are inclusive:

| Setting | OCTA | GSO |
| --- | --- | --- |
| Training `scene_min` / `scene_max` | `0` / `3099` | `0` / `1899` |
| Training `mask_root` | `None` | `None` |
| Validation `scene_min` / `scene_max` | `3100` / `3199` | `1900` / `1999` |
| Validation `mask_root` | `data/OCTScenes_grouped/128x128/GT_mask/3100-3199` | `data/GSO_new/128x128/GT_mask` |
| `--data_path` | `data/OCTScenes_grouped/128x128/image` | `data/GSO_new/128x128/image` |

Use `--data_name octa` for both recipes: it selects the model's RGB-CNN attention branch and names the log group. The selected dataset comes from the image root, scene bounds, and validation mask root. Set these constructor fields in the trainer before switching datasets.

### OCTA training

1. Place scene images under `image/<scene>/<frame>.png` for scenes 0–3199, and validation masks under `GT_mask/3100-3199/<scene>_<frame>.png`.
2. Set the training and validation constructors below inside the `if args.data_name == 'octa':` block, preserving that block's indentation:

```python
train_dataset = OCTScenesDataset(
    images_root=args.data_path,
    scene_min=0,
    scene_max=3099,
    mask_root=None,
    image_transform=transforms.ToTensor(),
)
val_dataset = OCTScenesDataset(
    images_root=args.data_path,
    scene_min=3100,
    scene_max=3199,
    mask_root="data/OCTScenes_grouped/128x128/GT_mask/3100-3199",
    image_transform=transforms.ToTensor(),
    mask_transform=transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.long()),
    ]),
)
```

3. Activate the main environment and launch 30 epochs of fresh training:

```bash
conda activate gaussian_grouping
cd /path/to/SA-OCL
source scripts/env.sh

torchrun --standalone --nproc_per_node 1 GOLD_Code/train_ocl.py \
    --data_name octa --data_path data/OCTScenes_grouped/128x128/image \
    --model_name octa_gold --checkpoint_path "" \
    --batch_size 8 --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8 \
    --seed 3496339326 --epochs 30 --num_cls 11 \
    --reg_start 0.1 --reg_final 0.1 --reg_steps 697500 \
    --kld_start 1 --kld_final 1 --kld_start_steps 0 --kld_final_steps 80000 \
    --tau_start 1.0 --tau_final 0.1 --tau_start_steps 100000 --tau_final_steps 300000 \
    --beta 100 --num_iterations 2
```

The trained state dictionary is `logs/octa/octa_gold/best_model.pt`. Set the OCTA Gaussian trainer and renderer's `--checkpoint_path` to this file.

To use [scripts/paper/gold/train_oct.sh](scripts/paper/gold/train_oct.sh), configure the same dataset block, set the launcher's `--data_path` to your OCTA image root and `--model_name` to your run name, and use `--checkpoint_path ""` for fresh training. For continued training, set that argument to the GOLD resume archive and choose the target total epoch count.

### GSO training

Set the dataset constructors to the GSO bounds and validation-mask root in the table, then run:

```bash
conda activate gaussian_grouping
cd /path/to/SA-OCL
source scripts/env.sh

torchrun --standalone --nproc_per_node 1 GOLD_Code/train_ocl.py \
    --data_name octa --data_path data/GSO_new/128x128/image \
    --model_name gso_gold --checkpoint_path "" \
    --batch_size 8 --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8 \
    --seed 3496339326 --epochs 40 --num_cls 11 \
    --reg_start 0.1 --reg_final 0.1 --reg_steps 570000 \
    --kld_start 1 --kld_final 1 --kld_start_steps 0 --kld_final_steps 80000 \
    --tau_start 1.0 --tau_final 0.1 --tau_start_steps 100000 --tau_final_steps 300000 \
    --beta 100 --num_iterations 2
```

This saves `logs/octa/gso_gold/best_model.pt`. For [scripts/paper/gold/train_gso.sh](scripts/paper/gold/train_gso.sh), set the same image root, model name, architecture arguments, and fresh/resume checkpoint choice.

### Checkpoints and resuming

Use **`torchrun` even for one GPU**, because the trainer initializes NCCL and distributed training. Keep the model dimensions explicit: `--num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8 --num_cls 11 --num_iterations 2`.

Outputs go to `logs/<data_name>/<model_name>/`:

| File | Purpose |
| --- | --- |
| `best_model.pt` | Validation-best plain state dictionary for Gaussian training/rendering |
| `best_model_until_<steps>_steps.pt` | Selected plain state dictionary within the `--steps` budget |
| `checkpoint.pt.tar` | Model, optimizer, epoch, and validation state for GOLD resume |

For fresh training, `--checkpoint_path ""` explicitly disables loading a resume file. To resume OCTA, repeat its training command with `--checkpoint_path logs/octa/octa_gold/checkpoint.pt.tar`; for GSO, use its corresponding archive. `--epochs` is the target total epoch count, not an additional number of epochs. Preserve the architecture settings and use the validation-best `best_model.pt` for the Gaussian pipeline.

For scheduler submission, configure the launcher environment initialization for your installation and submit from the project root:

```bash
mkdir -p logs/slurm
sbatch scripts/paper/gold/train_oct.sh
# Or submit the GSO launcher:
sbatch scripts/paper/gold/train_gso.sh
```

## Run the combined training and evaluation jobs

The metric launchers execute **Gaussian training → rendering → evaluation** per scene, consuming the frozen GOLD checkpoint trained above. Steps run sequentially inside each array task; tasks can run in parallel.

| Dataset | Launcher | Dataset root | GOLD checkpoint to configure |
| --- | --- | --- | --- |
| OCTA | [run_metric_ours_oct.sh](scripts/paper/evaluation/run_metric_ours_oct.sh) | `data/OCTScenes_grouped` | `logs/octa/octa_gold/best_model.pt` |
| GSO | [run_metric_ours_gso.sh](scripts/paper/evaluation/run_metric_ours_gso.sh) | `data/GSO_new` | `logs/octa/gso_gold/best_model.pt` |

Before launching:

1. Set dataset roots, scene-output paths, and both training/rendering `--checkpoint_path` arguments to your prepared data and GOLD checkpoint.
2. Set `START_SCENE`, `SCENES_PER_JOB`, and the array range to cover your evaluation scenes. For OCTA scenes 3100–3199, use `START_SCENE=3100`, `SCENES_PER_JOB=10`, and array `0-9`. For GSO, start at 1900 and select the scene set for your experiment. Each task processes `START_SCENE + task_id * SCENES_PER_JOB` through that value plus `SCENES_PER_JOB - 1`.
3. Set the Gaussian training budget explicitly. Use `--iterations 300000` for the paper's budget, and keep `--ours --train_split --eval --regloss3d`, the dataset flag (`--OCT` or `--GSO`), and the 122/6/128 model dimensions consistent.
4. Set a report basename for the run, e.g. `evaluation/reports/COLMAP/OCTA/ours/run_01`, and configure `--save` in every metric call whose scene outputs you want to keep.
5. Configure environment activation for your installation, then test one scene with the manual command below.

**Without `--save`, `metric.py` deletes the complete `--model_path` directory after evaluation.** Reports remain at the separate `--output` location. Apply the retention choice to each branch of an array launcher.

Submit the configured launchers from the project root:

```bash
cd /path/to/SA-OCL
mkdir -p logs logs/slurm
sbatch scripts/paper/evaluation/run_metric_ours_oct.sh
# Or run the GSO scene set:
sbatch scripts/paper/evaluation/run_metric_ours_gso.sh
```

For execution without a scheduler, use the manual pipeline below.

## Run one scene manually

This GSO example **keeps its output**. Run in the main environment on a GPU with complete GSO inputs and the trained checkpoint:

```bash
conda activate gaussian_grouping
cd /path/to/SA-OCL
source scripts/env.sh
SAOCL_SCENE=1900
SAOCL_GOLD=logs/octa/gso_gold/best_model.pt
SAOCL_OUTPUT=output/COLMAP/GSO_new/${SAOCL_SCENE}_ours

python training/paper/train_GSO.py \
    -s data/GSO_new -m "${SAOCL_OUTPUT}" \
    --config_file config/gaussian_dataset/train.json \
    --size 128 --scene_id "${SAOCL_SCENE}" --iterations 300000 \
    --GSO --ours --train_split --eval --regloss3d \
    --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8 --num_cls 11 \
    --checkpoint_path "${SAOCL_GOLD}" && \
python rendering/paper/render_ours_GSO.py \
    -m "${SAOCL_OUTPUT}" --GSO --ours \
    --num_slots 8 --slot_size 128 --int_size 122 --ext_size 6 --bck_size 8 --num_cls 11 \
    --checkpoint_path "${SAOCL_GOLD}" && \
python evaluation/paper/metric.py \
    --model_path "${SAOCL_OUTPUT}" \
    --output evaluation/reports/COLMAP/GSO_new/ours/manual \
    --test --ours --save
```

`&&` starts the next stage only after the preceding command succeeds. For OCTA, substitute:

| GSO setting | OCTA setting |
| --- | --- |
| `training/paper/train_GSO.py` | `training/paper/train_ours.py` |
| `rendering/paper/render_ours_GSO.py` | `rendering/paper/render_ours.py` |
| `-s data/GSO_new`, `--GSO` | `-s data/OCTScenes_grouped`, `--OCT` |
| Scene 1900 | Available held-out scene, e.g. 3100 |
| `logs/octa/gso_gold/best_model.pt` | `logs/octa/octa_gold/best_model.pt` |
| `output/COLMAP/GSO_new/...` | `output/COLMAP/OCTA/...` |
| GSO report base | Distinct base under `evaluation/reports/COLMAP/OCTA/ours/` |

The example sets **300,000 Gaussian iterations** to match the paper's budget. Set `--iterations <N>` explicitly for each experiment; omitting it uses the trainer default of 30,000. The final iteration is added to the save list. Rendering loads the latest saved point cloud, or `--iteration <N>` selects one. `--checkpoint_iterations <N> ...` saves resumable Gaussian archives; `--start_checkpoint <path>` resumes them.

## Outputs and metrics

```text
output/COLMAP/<dataset>/<scene>_ours/
├── cfg_args
├── input.ply
├── cameras.json
├── point_cloud/iteration_300000/point_cloud.ply
├── train/
│   ├── renders/                 rendered RGB
│   ├── gt/                      GT RGB
│   ├── objects_pred/            integer predicted IDs
│   ├── objects_rgb_pred/        colored ID visualization
│   ├── objects_feature16/       PCA feature visualization
│   ├── gt_ocl_mask/             integer GT IDs
│   ├── gt_objects_color/        GT visualization
│   ├── concat/                  comparison images
│   └── fps.txt
└── test/                        same groups for held-out views
```

`objects_feature16` contains PCA visualizations of the paper model's **122-dimensional features**. Use the integer ID images in `objects_pred` for segmentation evaluation. Render names such as `00000.png` index the rendered camera sequence; match each result to its exported GT image when scoring.

Evaluation always scores train views; `--test` also scores held-out views. `--ours` includes renderer FPS. Frame scores are averaged within each scene:

| Measure | Meaning |
| --- | --- |
| FG-ARI | Adjusted Rand Index on GT foreground pixels |
| ARI | Adjusted Rand Index including background |
| AMI-O / AMI-A | Adjusted Mutual Information on foreground / all pixels |
| mIoU | Foreground IoU with Hungarian matching of predicted/GT masks |
| PSNR / SSIM / LPIPS | RGB reconstruction quality |
| FPS | CUDA-event timing of the render call; excludes codebook assignment and disk I/O |

ARI, AMI, and matched IoU allow label permutations within a scene. They alone do not establish cross-scene identity consistency; that depends on the shared checkpoint/codebook and cross-scene comparisons. Held-out dataset scenes and held-out views within an optimized scene are different concepts.

`--output` is a **report basename**: `.../manual` produces `manual_train.txt` and, with `--test`, `manual_test.txt`. Reports append scene rows; use a new basename per run to avoid mixed/duplicate results. Array tasks share report files without explicit locking, so check coverage and duplicate rows before aggregation.

For example, configure OCTA's report base as `evaluation/reports/COLMAP/OCTA/ours/run_01` and GSO's as `evaluation/reports/COLMAP/GSO_new/ours/run_01`. The evaluator creates the corresponding `_train.txt` and `_test.txt` files.

Logs follow each launcher's `#SBATCH --output` path. [evaluation/reports/get_scores.ipynb](evaluation/reports/get_scores.ipynb) aggregates score text; run it with working directory `evaluation/reports/`, select the report paths for your run, and combine only results using the same experiment settings.

## Baselines and other components

These launchers also train, render, and evaluate per scene. They use the **16-feature core**, baseline classifiers, and prepared pseudo-masks:

| Launcher | Method / masks | Datasets |
| --- | --- | --- |
| [run_metric_gg.sh](scripts/paper/evaluation/run_metric_gg.sh) | Gaussian Grouping / DEVA | GSO |
| [run_metric_gg_sam2.sh](scripts/paper/evaluation/run_metric_gg_sam2.sh) | Gaussian Grouping / SAM2 tracking | OCTA, GSO |
| [run_metric_uni_sam.sh](scripts/paper/evaluation/run_metric_uni_sam.sh) | Unified Lift / SAM1 | OCTA, GSO |
| [run_metric_uni_sam2.sh](scripts/paper/evaluation/run_metric_uni_sam2.sh) | Unified Lift / independent SAM2 | OCTA, GSO |

Configure dataset paths, evaluation scenes, output/report locations, and `--save` for these jobs in the same way as the paper launchers. Build the 16-feature rasterizer, prepare the mask folders from [Data setup](#data-setup), and keep baseline classifier settings consistent between training and rendering. [scripts/original](scripts/original/) contains inherited training, preprocessing, and editing launchers.

Additional components:

- [GOLD_Code/train_vqvae.py](GOLD_Code/train_vqvae.py), [test_gold.py](GOLD_Code/test_gold.py), and `vqvae/` implement auxiliary image reconstruction. The paper Gaussian pipeline uses the trained object-centric backbone directly.
- [GOLD_Code/train_ocl_nobg.py](GOLD_Code/train_ocl_nobg.py) and [test_ocl.py](GOLD_Code/test_ocl.py) use the alternative `ocl.OCL` model. Set their dataset splits and checkpoint architecture for that model when running those experiments.
- [editing/original](editing/original/) contains object removal and LaMa-assisted inpainting. Configure input/output paths, object selection, and external inpainting model paths in the editing launchers.
- Original LERF rendering requires text/grounded segmentation dependencies, including GroundingDINO, and its dataset inputs.
- [evaluation/notebooks/ARI_test.ipynb](evaluation/notebooks/ARI_test.ipynb) supports additional analysis. Configure dataset/checkpoint paths and install its referenced `testing_models.contrasa_base` implementation before using it.

## Experiment configuration

[config/gaussian_dataset/train.json](config/gaussian_dataset/train.json) is loaded after Gaussian argument parsing and overrides the corresponding settings:

| Setting | Value |
| --- | --- |
| `densify_until_iter` | 10,000 |
| `num_classes` | 256; baseline classifier count |
| `reg3d_interval` | 5 |
| `reg3d_k` | 5 |
| `reg3d_lambda_val` | 2 |
| `reg3d_max_points` | 200,000 |
| `reg3d_sample_size` | 1,000 |

Set dataset splits using the GOLD table, the scene range using the metric launcher, and Gaussian iterations using `--iterations`. Keep the 122-dimensional intrinsic code, 6-dimensional extrinsic code, and 128-dimensional full slot consistent across stages.

`--regloss3d` enables neighborhood consistency. The standard trainer uses feature MSE and sharpens resized GOLD masks before constructing feature targets. Its neighborhood term compares distributions over the selected per-view codes plus background, and the helper scales KL by the probability-vector width. `--masked_pixel_loss` selects an alternative feature loss. Record these flags, the configuration file, scene list, and checkpoint with each experiment.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing `core` / `GOLD_Code` import | Activate the environment and source `scripts/env.sh` in Bash, or use `scripts/run.sh` |
| Rasterizer `_C` fails / undefined symbols | Match Python/PyTorch/CUDA ABI; rebuild the correct variant in place and inspect the loaded path |
| C++ runtime `GLIBCXX` error | Use a C++ runtime compatible with the compiler that built the extension; activate the environment before sourcing the helper |
| Missing `nvcc` / CUDA mismatch | Check the full toolkit, `CUDA_HOME`, `nvcc --version`, and `torch.version.cuda` |
| No kernel image for GPU | Rebuild for the GPU architecture used by training |
| GOLD checkpoint/feature shape mismatch | Match `ocl_origin.OCL`, prototype count, slots, and 122/6/128 dimensions; use plain `best_model.pt` for Gaussian stages |
| Missing distributed `RANK` | Use `torchrun` for GOLD, including single-GPU training |
| Empty GOLD loader / missing validation masks | Check image folders, inclusive scene bounds, and the dataset-specific `mask_root` in the trainer |
| Camera or image file not found | Match the dataset-root layout, archive member names, scene IDs, and frame filenames |
| Model disappears after scoring | Set metric `--save` in every applicable launcher branch |
| SAM2 Hydra config missing | Use the SAM2 environment and a package providing the SAM2.1 configuration |
| Missing GroundingDINO / `lpipsPyTorch` / notebook module | Install and configure the dependency required by that entrypoint |
| Scheduler cannot open a log file | Submit from the project root and create the configured log directory before submission |

## References

- Paper: [Scene-Agnostic Object-Centric Representation Learning for 3D Gaussian Splatting, arXiv:2604.09045v1](https://arxiv.org/abs/2604.09045v1).
- GOLD: [Learning global object-centric representations via disentangled slot attention](https://link.springer.com/article/10.1007/s10994-024-06687-9) and [official implementation](https://github.com/Tonglin-Chen/GOLD).
- Dataset: [OCTScenes](https://huggingface.co/datasets/Yinxuan/OCTScenes) and [dataset paper](https://arxiv.org/abs/2306.09682).
- Synthetic data generation: [Kubric](https://github.com/google-research/kubric).
- Inherited Gaussian core: [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping).
- Root [LICENSE](LICENSE) and component licenses apply to their respective code; consult upstream package/model terms when obtaining assets.
