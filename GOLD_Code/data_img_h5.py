import torch
import os
from torchvision import transforms 
from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision.utils import save_image
from PIL import Image
from glob import glob
from typing import Optional
class GlobVideoDataset(Dataset):
    def __init__(self, root, phase, data_name):
        self.root = root
        self.data_name = data_name
        with h5py.File(root, 'r', libver='latest', swmr=True) as f:
            self.total_data =f[phase]
            
            self.video = self.total_data['image'][()]
            self.seg = self.total_data['segment'][()]
            self.clss = self.total_data['cls'][()]

            if self.data_name == 'unique_gso':
                num_video, num_view, _, _, _ = self.video.shape
                self.video = self.video[:, 0]
                self.seg = self.seg[:, 0]

            
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):

        img = self.transform(self.video[idx])
        seg = torch.from_numpy(self.seg[idx]).to(torch.int64)
        clss = self.clss[idx]


        return img, seg, clss
import os
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OCTScenesDataset(Dataset):
    """
    Expects images like:
      images/{scene}/{scene}_{frame}.png

    Optionally loads masks like:
      GT_mask/4900-4999/{scene}_{frame}.png
    """
    def __init__(
        self,
        images_root: str,
        scene_min: int,
        scene_max: int,
        mask_root:  Optional[str] = None,
        image_transform=None,
        mask_transform=None,
        return_paths: bool = False,
    ):
        self.images_root = images_root
        self.scene_min = scene_min
        self.scene_max = scene_max
        self.mask_root = mask_root
        self.return_paths = return_paths

        self.image_transform = transforms.Compose([                
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

    ])
        # masks are usually single-channel; keep them as integer labels if needed
        self.mask_transform = mask_transform  # can be None

        # Grab all images in all scene subfolders
        all_image_paths = sorted(glob(os.path.join(images_root, "*", "*")))

        # Keep only those whose scene is in [scene_min, scene_max]
        self.samples = []
        for p in all_image_paths:
            scene_str, frame_str = self._parse_scene_frame(p)  # e.g. "4903", "0012"
            scene_id = int(scene_str)
            if scene_min <= scene_id <= scene_max:
                self.samples.append((scene_str, frame_str, p))
    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _parse_scene_frame(img_path: str):
        # img_path: .../images/{scene}/{scene}_{frame}.png

        filename = os.path.basename(img_path)          # {scene}_{frame}.png
        stem = os.path.splitext(filename)[0]           # {scene}_{frame}
        scene_str = os.path.basename(os.path.dirname(img_path))  
        frame_str = stem     # split once
        return scene_str, frame_str

    def __getitem__(self, idx):
        scene_str, frame_str, img_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.image_transform(img) if self.image_transform else img

        if self.mask_root is None:
            if self.return_paths:
                return img, img_path
            return img

        # Mask file is named {scene}_{frame}.png
        mask_path = os.path.join(self.mask_root, f"{scene_str}_{frame_str}.png")

        
        mask = Image.open(mask_path).convert("L")  # or keep as-is if already single channel

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            # default: convert to tensor without normalizing to [0,1] labels
            # If your mask is 0/255 binary, this makes it 0/255. You can threshold later.
            mask = torch.from_numpy(
                __import__("numpy").array(mask)
            ).to(torch.int64)

        if self.return_paths:
            return img, mask, img_path, mask_path
        return img, mask
    

class ReplicaDataset(Dataset):
    """
    Automatically crawls all scene folders in the root directory.
    Format:
      root/{scene}/image/s{frame}.png
      root/{scene}/masks/s{frame}.png
    """
    def __init__(
        self,
        root: str,
        image_transform=None,
        mask_transform=None,
        mask_ = False,
        return_paths: bool = False,
    ):
        self.root = root
        self.return_paths = return_paths
        self.mask_ =mask_
        # Default transform if none provided
        self.image_transform = image_transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        self.mask_transform = mask_transform

        self.samples = []
        
        # 1. Get all directories in root (each represents a scene)
        # Using sorted(os.listdir) to ensure deterministic ordering
        scenes = sorted([
            d for d in os.listdir(root) 
            if os.path.isdir(os.path.join(root, d))
        ])

        # 2. Iterate through every scene and find images
        for scene in scenes:
            img_dir = os.path.join(root, scene, "images")
            
            # Find all png files in the image directory
            images = sorted(glob(os.path.join(img_dir, "*.png")))
            
            for p in images:
                # frame_str will be 's0', 's1', etc.
                frame_str = os.path.splitext(os.path.basename(p))[0]
                self.samples.append((scene, frame_str, p))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_str, frame_str, img_path = self.samples[idx]

        # Load Image
        img = Image.open(img_path).convert("RGB")
        if self.image_transform:
            img = self.image_transform(img)

        # Construct Mask Path
        # root/{scene}/masks/s{frame}.png
        mask_path = os.path.join(self.root, scene_str, "masks", f"{frame_str}.png")
        
        # Check if mask exists; if not, return just the image (or handle error)
        if not self.mask_:
            if self.return_paths:
                return img, img_path
            return img

        # Load Mask
        mask = Image.open(mask_path).convert("L")

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            # Standard conversion for segmentation: Numpy array to LongTensor
            mask = torch.from_numpy(__import__("numpy").array(mask)).to(torch.long)

        if self.return_paths:
            return img, mask, img_path, mask_path
        
        return img, mask

# ---- build the splits ----


