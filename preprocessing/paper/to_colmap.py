
import argparse
import os
import sys

# Add the current directory to path to ensure imports work
sys.path.append(os.getcwd())

# Import the loaders from the code you provided
# Assuming your file is located at scene/dataset_readers.py
try:
    from core.original.scene_16.dataset_readers import readOCTSceneInfo, readGSOSceneInfo
except ImportError:
    print("Error: Could not import dataset_readers.")
    print("Please ensure the code you provided is saved in 'core/original/scene_16/dataset_readers.py' or adjust the import.")
    sys.exit(1)

def pre_generate_oct(args):
    """
    Iterates through OCT scenes and triggers generation.
    """
    print(f"\n[OCTScenes] Processing scenes")
    
    for scene_id in range(3100,3200):
        print(f"--- Processing Scene ID: {scene_id} ---")

            # We call the loader with random_init=True. 
            # In your code, this triggers 'generate_point_cloud_from_known_poses'
            # and saves the resulting PLY to disk.
        readOCTSceneInfo(
                path=args.source_path,
                size=args.size,
                eval=False,            # False ensures we use all available images for reconstruction
                scene_id=int(scene_id), # Your code expects an integer for formatting {:04d}
                unilift=False,
                ours=False,
                sam2=False,
                random_init=True,       # <--- THIS IS THE KEY
                train_split=False
            )
        print(f"Successfully generated Colmap data for Scene {scene_id}")


def pre_generate_gso(args):
    """
    Iterates through GSO scenes and triggers generation.
    """
    print(f"\n[GSO] Processing scenes")
    
    for scene_id in range(1900,2000):
        try:
            print(f"--- Processing Scene ID: {scene_id} ---")

            readGSOSceneInfo(
                        path=args.source_path,
                        size=args.size,
                        eval=False,
                        scene_id=int(scene_id),
                        unilift=False,
                        ours=False,
                        sam2=False,
                        random_init=True,       # <--- THIS IS THE KEY
                        train_split=False
                    )
            print(f"Successfully generated Colmap data for Scene {scene_id}")
        except:
            print("No match, skipped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate Colmap Point Clouds for Gaussian Grouping")
    
    parser.add_argument("--type", type=str, required=True, choices=["OCT", "GSO"], 
                        help="Dataset type to process")
    parser.add_argument("--source_path", type=str, required=True, 
                        help="Root path to the dataset (e.g., .../OCTScenes_grouped/128x128)")
    parser.add_argument("--size", type=int, default=128, 
                        help="Image resolution size (default: 128)")
    
    args = parser.parse_args()

    if args.type == "OCT":
        pre_generate_oct(args)
    elif args.type == "GSO":
        pre_generate_gso(args)
    
    print("\nProcessing complete.")