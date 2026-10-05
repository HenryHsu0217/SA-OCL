# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import re
import sys
from PIL import Image
from typing import NamedTuple
from core.paper.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from core.original.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import pycolmap
import sqlite3
import json
import shutil
from pathlib import Path
from plyfile import PlyData, PlyElement
from core.original.utils.sh_utils import SH2RGB
from core.paper.scene.gaussian_model import BasicPointCloud
import tarfile
import random
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    objects: np.array

class OCTCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    objects: np.array
    gt_ocl_mask: np.array
    CX: np.array
    CY: np.array
    fx: np.array
    fy: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def rotmat2qvec(R,GSO=False):
    """
    Converts a 3x3 rotation matrix to a unit quaternion (w, x, y, z).
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[:, np.argmax(vals)]
    if q[3] < 0 and GSO:
        q *= -1
    # Reorder to (w, x, y, z) as COLMAP expects
    return np.array([q[3], q[0], q[1], q[2]])
def write_colmap_text_files(output_dir, cameras, images):
    """
    Writes standard COLMAP text files to disk.
    This bypasses pycolmap API version conflicts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Write cameras.txt
    with open(os.path.join(output_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam_id, cam_data in cameras.items():
            model, w, h, params = cam_data
            params_str = " ".join(map(str, params))
            f.write(f"{cam_id} {model} {w} {h} {params_str}\n")

    # 2. Write images.txt
    with open(os.path.join(output_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id, img_data in images.items():
            qvec, tvec, cam_id, name = img_data
            q_str = " ".join(map(str, qvec))
            t_str = " ".join(map(str, tvec))
            f.write(f"{img_id} {q_str} {t_str} {cam_id} {name}\n")
            f.write("\n") # Empty line for 0 2D points (triangulator will fill this)

    # 3. Write empty points3D.txt (required for loader)
    with open(os.path.join(output_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
def generate_point_cloud_from_known_poses(cam_infos, image_path, output_path,GSO=False):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    database_path = output_path / "database.db"
    manual_input_dir = output_path / "manual_input"
    
    # 1. Clean up old files to ensure fresh start
    if database_path.exists():
        os.remove(database_path)
    if manual_input_dir.exists():
        shutil.rmtree(manual_input_dir)

    print("1. Extracting features (SIFT)...")
    # This assigns IDs to images and stores features in the DB
    pycolmap.extract_features(database_path, image_path)
    
    print("2. Matching features...")
    pycolmap.match_exhaustive(database_path)

    print("3. Syncing IDs and writing manual input...")
    
    # Fetch the Image IDs that COLMAP assigned in step 1
    name_to_id = {}
    db = sqlite3.connect(str(database_path))
    cursor = db.cursor()
    cursor.execute("SELECT name, image_id FROM images")
    for name, image_id in cursor.fetchall():
        name_to_id[name] = image_id
    db.close()

    # Prepare data for writing
    cameras_dict = {}
    images_dict = {}
    
    # Define Camera 1 (Assuming all use same intrinsics)
    first_cam = cam_infos[0]
    camera_id = 1
    cameras_dict[camera_id] = (
        "PINHOLE", 
        first_cam.width, 
        first_cam.height, 
        [first_cam.fx, first_cam.fy, first_cam.CX, first_cam.CY]
    )

    count = 0
    for cam in cam_infos:
        image_name = os.path.basename(cam.image_path)
        
        # We can only use images that feature extraction actually found
        if image_name not in name_to_id:
            print(f"Skipping {image_name} (not found in feature database)")
            continue

        db_id = name_to_id[image_name]
        qvec = rotmat2qvec(cam.R,GSO)
        tvec = cam.T.flatten()
        
        images_dict[db_id] = (qvec, tvec, camera_id, image_name)
        count += 1

    # Write the files to disk (This works on ALL versions)
    write_colmap_text_files(manual_input_dir, cameras_dict, images_dict)

    print(f"4. Loading reconstruction ({count} images) and Triangulating...")
    
    # Load the "empty" reconstruction from the text files we just wrote
    recon = pycolmap.Reconstruction(manual_input_dir)

    # Run Triangulation
    # This takes the poses from 'recon' and the features from 'database_path'
    # and fills in the 3D points.
    pycolmap.triangulate_points(recon, database_path, image_path, output_path)

    # Save final result (bin files) for ObjectGS
    recon.write(output_path)
    
    print(f"SfM Point Cloud generated at {output_path}")
    return output_path
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, objects_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path) if os.path.exists(image_path) else None
        object_path = os.path.join(objects_folder, image_name + '.png')
        objects = Image.open(object_path) if os.path.exists(object_path) else None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, objects=objects)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, object_path, llffhold=8, n_views=100, random_init=False, train_split=False):
    try:
        cameras_extrinsic_file = os.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    object_dir = 'object_mask' if object_path == None else object_path
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), objects_folder=os.path.join(path, object_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            train_cam_infos = []
            test_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
                else:
                    test_cam_infos.append(cam_info)

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

            if n_views == 100:
                pass 
            elif n_views == 50:
                idx_sub = np.linspace(0, len(train_cam_infos)-1, round(len(train_cam_infos)*0.5)) # 50% views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
            elif isinstance(n_views,int):
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views) # 3views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
                print(train_cam_infos)
            else:
                raise NotImplementedError
        print("Training images:     ", len(train_cam_infos))
        print("Testing images:     ", len(test_cam_infos))

    else:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            train_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
            test_cam_infos = []
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if random_init:
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
        ply_path = os.path.join(path, "sparse/0/points3D_randinit.ply")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readOCTSceneCameras(img_path, intrinsic_path, extrinsic_path, scene_id, mask_path, size, ours, gt_ocl_mask_path):
    cam_infos = []

    intr_pat = re.compile(rf"^bbox/{scene_id:04d}/{scene_id:04d}_(\d+)\.txt$")
    extr_pat = re.compile(rf"^pose/{scene_id:04d}_(\d+)\.txt$")

    # Open both tars and keep them open while reading
    with tarfile.open(intrinsic_path, "r:*") as intr_tar, \
         tarfile.open(extrinsic_path, "r:*") as extr_tar:

        # Build frame_id -> TarInfo maps
        intr_members = {}
        for m in intr_tar.getmembers():
            mm = intr_pat.match(m.name)
            if mm:
                fid = int(mm.group(1))
                intr_members[fid] = m

        extr_members = {}
        for m in extr_tar.getmembers():
            mm = extr_pat.match(m.name)
            if mm:
                fid = int(mm.group(1))
                extr_members[fid] = m

        # Use only frames that exist in BOTH
        frame_ids = sorted(set(intr_members.keys()) & set(extr_members.keys()))
        for fid in frame_ids:
            height = size
            width = size

            # Read text from tar members
            extr_txt = extr_tar.extractfile(extr_members[fid]).read().decode("utf-8")
            intr_txt = intr_tar.extractfile(intr_members[fid]).read().decode("utf-8")

            intrinsic = np.array([list(map(float, ln.split()))
                                  for ln in intr_txt.strip().splitlines()])
            extrinsic = np.array([list(map(float, ln.split()))
                                  for ln in extr_txt.strip().splitlines()])

            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]

            R = extrinsic[:3, :3]
            T = extrinsic[:3, 3]

            FovY = focal2fov(fy, height)
            FovX = focal2fov(fx, width)

            image_path = os.path.join(img_path, f"{scene_id:04d}/{fid:02d}.png")
            image = Image.open(image_path) if os.path.exists(image_path) else None
            if not ours:
                mask_pathes = os.path.join(mask_path, f"{fid:02d}.png")
            else:
                mask_pathes = os.path.join(gt_ocl_mask_path, f"{scene_id}_{fid:02d}.png")
            mask = np.array(Image.open(mask_pathes).convert("L"), dtype=np.int64) if os.path.exists(mask_pathes) else None
            if ours:
                cam_info = OCTCameraInfo(
                    uid=fid, R=R, T=T, FovY=FovY, FovX=FovX,
                    image=image, image_path=image_path,
                    image_name=fid,
                    width=width, height=height, objects=mask, gt_ocl_mask=None,CX=cx, CY=cy,fx=fx, fy=fy
                )
            else:
                ocl_mask_path = os.path.join(gt_ocl_mask_path, f"{scene_id}_{fid:02d}.png")
                gt_ocl = np.array(Image.open(ocl_mask_path), dtype=np.int64) if os.path.exists(mask_pathes) else None
                cam_info = OCTCameraInfo(
                    uid=fid, R=R, T=T, FovY=FovY, FovX=FovX,
                    image=image, image_path=image_path,
                    image_name=fid,
                    width=width, height=height, objects=mask, gt_ocl_mask=gt_ocl,CX=cx, CY=cy,fx=fx, fy=fy
                )
            cam_infos.append(cam_info)
    return cam_infos

def readOCTSceneInfo(path, size, eval, scene_id, unilift = False, ours=False, sam2=False, llffhold=8, n_views=60, random_init=False, train_split=False, split_factor=1):
    """
    path: directory containing data/OCTScenes_grouped/128x128/
    Example:
        path = ".../OCTScenes_grouped/128x128"
        scene_id = "0000"

    This loader builds:
        - CameraInfo list
        - Nerf-style normalization
        - Random initial point cloud (like Blender loader)
    """
    intrinsic_path = os.path.join(path,f"{size}x{size}/intrinsics.tar")
    extrinsic_path = os.path.join(path,"poses.tar")
    img_path = os.path.join(path,f"{size}x{size}/image")
    if not unilift and not ours:
        mask_path = os.path.join(path,f"{size}x{size}/mask/{scene_id:04d}")
    elif unilift:
        mask_path = os.path.join(path,f"{size}x{size}/SAM_mask_gray/{scene_id:04d}")
    elif sam2:
        mask_path = os.path.join(path,f"{size}x{size}/SAM2_mask_8_point/{scene_id:04d}")
    else:
        mask_path = None
    gt_ocl_mask_path = os.path.join(path,f"{size}x{size}/GT_mask/3100-3199")

    cam_infos_unsorted = readOCTSceneCameras(img_path,intrinsic_path,extrinsic_path, scene_id, mask_path, size, ours, gt_ocl_mask_path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    if eval:
        if train_split:
            train_dir = os.path.join(img_path, f"{scene_id:04d}")
            train_names = sorted(os.listdir(train_dir))
            train_names = [
                int(f.split('.')[0])
                for f in train_names
                if os.path.isfile(os.path.join(train_dir, f))
            ]

            train_names = random.sample([train_name for train_name in train_names],k=int(len(cam_infos) * split_factor))
            print(sorted(train_names))
            train_cam_infos = []
            test_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
                else:
                    test_cam_infos.append(cam_info)

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

            if n_views == 100:
                pass 
            elif n_views == 50:
                idx_sub = np.linspace(0, len(train_cam_infos)-1, round(len(train_cam_infos)*0.5)) # 50% views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
            elif isinstance(n_views,int):
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views) # 3views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
    
            else:
                raise NotImplementedError
        print("Training images:     ", len(train_cam_infos))
        print("Testing images:     ", len(test_cam_infos))

    else:
        if train_split:
            train_dir = os.path.join(img_path, f"{scene_id:04d}")
            train_names = sorted(os.listdir(train_dir))
            train_names = [
                int(f.split('.')[0])
                for f in train_names
                if os.path.isfile(os.path.join(train_dir, f))
            ]
            train_names = random.sample([train_name for train_name in train_names],k=int(len(cam_infos) * split_factor))
            train_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
            test_cam_infos = []
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if random_init:
        # Since this data set has no colmap data, we start with random points
        print("Generating Point Cloud from known poses...")
        
        sfm_output_dir = os.path.join(path, f"{size}x{size}/colmap_triangulated/{scene_id:04d}")
        
        # Run the triangulation
        generate_point_cloud_from_known_poses(train_cam_infos, os.path.join(img_path, f"{scene_id:04d}"), sfm_output_dir)
        
        # Now point your ObjectGS code to this new folder
        # The triangulator outputs standard cameras.bin, images.bin, points3D.bin
        ply_path = os.path.join(path, f"{size}x{size}/points/{scene_id}_points3D.ply")
        
        xyz, rgb, _ = read_points3D_binary(os.path.join(sfm_output_dir, "points3D.bin"))
        storePly(ply_path, xyz, rgb)
        shutil.rmtree(sfm_output_dir)


    else:
        ply_path = os.path.join(path, f"{size}x{size}/points/{scene_id}_points3D.ply")
        bin_path = os.path.join(path, f"{size}x{size}/points/{scene_id}_points3D.bin")
        txt_path = os.path.join(path, f"{size}x{size}/points/{scene_id}_points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readGSOSceneCameras(img_path, intrinsic_path, extrinsic_path, scene_id, mask_path, size, ours, gt_ocl_mask_path):
    cam_infos = []

    extr_pat = re.compile(rf"^pose/{scene_id:04d}_(\d+)\.txt$")
    intrinsic = np.loadtxt(intrinsic_path)
    # Open both tars and keep them open while reading
    with tarfile.open(extrinsic_path, "r:*") as extr_tar:

        extr_members = {}
        for m in extr_tar.getmembers():
            mm = extr_pat.match(m.name)
            if mm:
                fid = int(mm.group(1))
                extr_members[fid] = m

        # Use only frames that exist in BOTH
        frame_ids = sorted(extr_members.keys())
        for fid in frame_ids:
            height = size
            width = size

            # Read text from tar members
            extr_txt = extr_tar.extractfile(extr_members[fid]).read().decode("utf-8")
            
            extrinsic = np.array([list(map(float, ln.split()))
                                  for ln in extr_txt.strip().splitlines()])

            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]

            R = extrinsic[:3, :3]
            T = extrinsic[:3, 3]

            # Convert coordinate system from OpenGL (Blender) to OpenCV (Colmap/3DGS)
            # OpenGL: Right: +X, Up: +Y, Forward: -Z
            # OpenCV: Right: +X, Up: -Y, Forward: +Z
            # We must flip the Y and Z axes of the Camera Frame.
            
            # 1. Flip the Y and Z rows of the Rotation Matrix
            R[1, :] *= -1
            R[2, :] *= -1
            
            # 2. Flip the Y and Z components of the Translation Vector
            T[1] *= -1
            T[2] *= -1

            FovY = focal2fov(fy, height)
            FovX = focal2fov(fx, width)

            image_path = os.path.join(img_path, f"{scene_id:04d}/{fid:02d}.png")
            image = Image.open(image_path) if os.path.exists(image_path) else None
            if not ours:
                mask_pathes = os.path.join(mask_path, f"{fid:02d}.png")
            else:
                mask_pathes = os.path.join(gt_ocl_mask_path, f"{scene_id}_{fid:02d}.png")
            mask = np.array(Image.open(mask_pathes), dtype=np.int64) if os.path.exists(mask_pathes) else None
            if ours:
                cam_info = OCTCameraInfo(
                    uid=fid, R=R, T=T, FovY=FovY, FovX=FovX,
                    image=image, image_path=image_path,
                    image_name=fid,
                    width=width, height=height, objects=mask, gt_ocl_mask=None, CX=cx, CY=cy,fx=fx, fy=fy
                )
            else:
                ocl_mask_path = os.path.join(gt_ocl_mask_path, f"{scene_id}_{fid:02d}.png")
                gt_ocl = np.array(Image.open(ocl_mask_path), dtype=np.int64) if os.path.exists(mask_pathes) else None
                cam_info = OCTCameraInfo(
                    uid=fid, R=R, T=T, FovY=FovY, FovX=FovX,
                    image=image, image_path=image_path,
                    image_name=fid,
                    width=width, height=height, objects=mask, gt_ocl_mask=gt_ocl,CX=cx, CY=cy,fx=fx, fy=fy
                )
            cam_infos.append(cam_info)
    return cam_infos

def readGSOSceneInfo(path, size, eval, scene_id, unilift = False, ours=False, sam2=False, llffhold=8, n_views=60, random_init=False, train_split=False, split_factor=1):
    """
    path: directory containing data/OCTScenes_grouped/128x128/
    Example:
        path = ".../OCTScenes_grouped/128x128"
        scene_id = "0000"

    This loader builds:
        - CameraInfo list
        - Nerf-style normalization
        - Random initial point cloud (like Blender loader)
    """
    intrinsic_path = os.path.join(path,"intrinsic.txt")
    extrinsic_path = os.path.join(path,"poses.tar")
    img_path = os.path.join(path,f"{size}x{size}/image")
    if not unilift and not ours and not sam2:
        mask_path = os.path.join(path,f"{size}x{size}/DEVA_mask/{scene_id:04d}")
    elif unilift:
        mask_path = os.path.join(path,f"{size}x{size}/SAM_mask/{scene_id:04d}")
    elif sam2:
        mask_path = os.path.join(path,f"{size}x{size}/SAM2_mask/{scene_id:04d}")
    else:
        mask_path = None
    gt_ocl_mask_path = os.path.join(path,f"{size}x{size}/GT_mask")

    cam_infos_unsorted = readGSOSceneCameras(img_path,intrinsic_path,extrinsic_path, scene_id, mask_path, size, ours, gt_ocl_mask_path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    if eval:
        if train_split:
            train_dir = os.path.join(img_path, f"{scene_id:04d}")
            train_names = sorted(os.listdir(train_dir))
            train_names = [
                int(f.split('.')[0])
                for f in train_names
                if os.path.isfile(os.path.join(train_dir, f))
            ]
            train_names = random.sample([train_name for train_name in train_names],k=int(len(cam_infos) * split_factor))
            train_cam_infos = []
            test_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
                else:
                    test_cam_infos.append(cam_info)

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

            if n_views == 100:
                pass 
            elif n_views == 50:
                idx_sub = np.linspace(0, len(train_cam_infos)-1, round(len(train_cam_infos)*0.5)) # 50% views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
            elif isinstance(n_views,int):
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views) # 3views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
    
            else:
                raise NotImplementedError
        print("Training images:     ", len(train_cam_infos))
        print("Testing images:     ", len(test_cam_infos))

    else:
        if train_split:
            train_dir = os.path.join(img_path, f"{scene_id:04d}")
            train_names = sorted(os.listdir(train_dir))
            train_names = [
                int(f.split('.')[0])
                for f in train_names
                if os.path.isfile(os.path.join(train_dir, f))
            ]
            train_names = random.sample([train_name for train_name in train_names],k=int(len(cam_infos) * split_factor))
            train_cam_infos = []
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
            test_cam_infos = []
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if random_init:
        # Since this data set has no colmap data, we start with random points
        print("Generating Point Cloud from known poses...")
        
        sfm_output_dir = os.path.join(path, f"{size}x{size}/colmap_triangulated/{scene_id:04d}")
        
        # Run the triangulation
        generate_point_cloud_from_known_poses(train_cam_infos, os.path.join(img_path, f"{scene_id:04d}"), sfm_output_dir)
        
        # Now point your ObjectGS code to this new folder
        # The triangulator outputs standard cameras.bin, images.bin, points3D.bin
        ply_path = os.path.join(path, f"{size}x{size}/points/{scene_id}_points3D.ply")
        xyz, rgb, _ = read_points3D_binary(os.path.join(sfm_output_dir, "points3D.bin"))
        storePly(ply_path, xyz, rgb)
        shutil.rmtree(sfm_output_dir)
    else:
        ply_path = os.path.join(path, f"{size}x{size}/points/{scene_id}_points3D.ply")
        bin_path = os.path.join(path, f"{size}x{size}/points/{scene_id}_points3D.bin")
        txt_path = os.path.join(path, f"{size}x{size}/points/{scene_id}_points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "OCTScenes": readOCTSceneInfo,
    "GSO":readGSOSceneInfo
}
