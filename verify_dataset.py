"""
Script to verify that LFW images match the existing dataset_description.pkl file.

This script:
1. Loads the existing dataset_description.pkl
2. Checks that image files exist and match the paths in the pickle
3. Reports any missing or mismatched images

Usage:
    python verify_dataset.py --lfw_dir ./LFWA+/lfw --pkl_file dataset_description.pkl
"""

import os
import pickle
import numpy as np
import argparse
from collections import Counter


def verify_dataset(lfw_dir, pkl_file):
    """
    Verify that images in the directory match the pickle file.
    
    Args:
        lfw_dir: Directory containing LFW images
        pkl_file: Path to dataset_description.pkl
    """
    print("="*60)
    print("VERIFYING LFW+A DATASET SETUP")
    print("="*60)
    
    # Load pickle file
    print(f"\nLoading {pkl_file}...")
    if not os.path.exists(pkl_file):
        print(f"ERROR: {pkl_file} not found!")
        return False
    
    with open(pkl_file, 'rb') as f:
        dc = pickle.load(f)
    
    print(f"✓ Loaded pickle file")
    print(f"  - Images in pickle: {len(dc['image_list'])}")
    print(f"  - Attributes shape: {dc['attributes'].shape}")
    print(f"  - Latent vars shape: {dc['latent_vars'].shape}")
    
    # Check directory exists
    print(f"\nChecking image directory: {lfw_dir}")
    if not os.path.exists(lfw_dir):
        print(f"ERROR: Directory {lfw_dir} not found!")
        print(f"Please create it and copy LFW images there.")
        return False
    
    print(f"✓ Directory exists")
    
    # Check images
    print(f"\nVerifying images...")
    image_list = dc['image_list']
    missing_images = []
    existing_images = []
    
    for i, img_path in enumerate(image_list):
        full_path = os.path.join(lfw_dir, img_path)
        if os.path.exists(full_path):
            existing_images.append(img_path)
        else:
            missing_images.append(img_path)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Checked {i+1}/{len(image_list)} images...")
    
    # Report results
    print(f"\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    print(f"Total images in pickle: {len(image_list)}")
    print(f"Images found on disk: {len(existing_images)} ({100*len(existing_images)/len(image_list):.1f}%)")
    print(f"Missing images: {len(missing_images)} ({100*len(missing_images)/len(image_list):.1f}%)")
    
    if len(missing_images) == 0:
        print("\n✓ SUCCESS: All images found!")
        return True
    else:
        print(f"\n⚠️  WARNING: {len(missing_images)} images are missing")
        
        # Show sample missing images
        if len(missing_images) <= 20:
            print("\nMissing images:")
            for img in missing_images[:20]:
                print(f"  - {img}")
        else:
            print("\nSample missing images (first 10):")
            for img in missing_images[:10]:
                print(f"  - {img}")
            print(f"  ... and {len(missing_images) - 10} more")
        
        # Check directory structure
        print("\nChecking directory structure...")
        sample_missing = missing_images[0] if missing_images else None
        if sample_missing:
            expected_dir = os.path.dirname(sample_missing)
            expected_full_dir = os.path.join(lfw_dir, expected_dir)
            if not os.path.exists(expected_full_dir):
                print(f"  ⚠️  Expected directory not found: {expected_full_dir}")
                print(f"  The directory structure might not match the pickle file.")
            else:
                print(f"  ✓ Directory exists: {expected_full_dir}")
                # List what's actually in there
                actual_files = os.listdir(expected_full_dir)
                print(f"  Files in directory: {len(actual_files)}")
                if len(actual_files) > 0:
                    print(f"  Sample files:")
                    for f in actual_files[:5]:
                        print(f"    - {f}")
        
        return False


def check_directory_structure(lfw_dir, pkl_file):
    """
    Check if the directory structure matches what's expected.
    """
    print("\n" + "="*60)
    print("DIRECTORY STRUCTURE CHECK")
    print("="*60)
    
    with open(pkl_file, 'rb') as f:
        dc = pickle.load(f)
    
    # Analyze path patterns
    image_list = dc['image_list']
    path_patterns = []
    for img_path in image_list[:100]:  # Sample first 100
        parts = img_path.split('/')
        if len(parts) == 2:
            path_patterns.append('person_name/person_name_XXXX.jpg')
        elif len(parts) == 1:
            path_patterns.append('person_name_XXXX.jpg')
        else:
            path_patterns.append('other')
    
    pattern_counts = Counter(path_patterns)
    print("\nPath patterns found in pickle file:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} samples")
    
    # Check actual directory structure
    print(f"\nActual directory structure in {lfw_dir}:")
    if os.path.exists(lfw_dir):
        subdirs = [d for d in os.listdir(lfw_dir) 
                  if os.path.isdir(os.path.join(lfw_dir, d))]
        files = [f for f in os.listdir(lfw_dir) 
                if os.path.isfile(os.path.join(lfw_dir, f))]
        
        print(f"  Subdirectories: {len(subdirs)}")
        print(f"  Files in root: {len(files)}")
        
        if len(subdirs) > 0:
            print(f"  Sample subdirectories:")
            for d in subdirs[:5]:
                print(f"    - {d}")
                subdir_path = os.path.join(lfw_dir, d)
                subdir_files = os.listdir(subdir_path)
                print(f"      Contains {len(subdir_files)} files")
                if len(subdir_files) > 0:
                    print(f"      Sample: {subdir_files[0]}")


def main():
    parser = argparse.ArgumentParser(description='Verify LFW+A dataset setup')
    parser.add_argument('--lfw_dir', type=str, default='./LFWA+/lfw',
                       help='Directory containing LFW images')
    parser.add_argument('--pkl_file', type=str, default='dataset_description.pkl',
                       help='Path to dataset_description.pkl file')
    parser.add_argument('--check_structure', action='store_true',
                       help='Also check directory structure')
    
    args = parser.parse_args()
    
    success = verify_dataset(args.lfw_dir, args.pkl_file)
    
    if args.check_structure:
        check_directory_structure(args.lfw_dir, args.pkl_file)
    
    if success:
        print("\n" + "="*60)
        print("✓ Dataset is ready to use!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️  Please fix missing images before running the code")
        print("="*60)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

