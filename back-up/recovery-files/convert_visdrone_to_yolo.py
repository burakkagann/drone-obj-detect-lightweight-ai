#!/usr/bin/env python3
"""
VisDrone to YOLO Format Converter
Converts VisDrone annotation format to YOLO format for training

VisDrone format: x,y,w,h,score,object_category,truncation,occlusion
YOLO format: class x_center y_center width height (normalized)
"""

import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

def convert_visdrone_box_to_yolo(img_size, box):
    """
    Convert VisDrone bounding box to YOLO format
    
    Args:
        img_size: (width, height) of image
        box: [x, y, w, h] in VisDrone format (absolute pixels)
    
    Returns:
        [x_center, y_center, width, height] in YOLO format (normalized 0-1)
    """
    img_w, img_h = img_size
    x, y, w, h = box
    
    # Convert to YOLO format (center coordinates, normalized)
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    
    return [x_center, y_center, width, height]

def convert_visdrone_annotations(dataset_dir):
    """
    Convert VisDrone annotations to YOLO format for a dataset split
    
    Args:
        dataset_dir: Path to dataset directory (e.g., VisDrone2019-DET-train)
    """
    dataset_dir = Path(dataset_dir)
    annotations_dir = dataset_dir / 'annotations'
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    
    if not annotations_dir.exists():
        print(f"Warning: {annotations_dir} not found. Skipping conversion.")
        return
    
    if not images_dir.exists():
        print(f"Warning: {images_dir} not found. Skipping conversion.")
        return
    
    # Create labels directory
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # VisDrone class mapping (1-indexed in VisDrone, 0-indexed in YOLO)
    # VisDrone classes: 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van, 
    #                  6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor
    # Class 0 in VisDrone means "ignored region" - we skip these
    
    annotation_files = list(annotations_dir.glob('*.txt'))
    print(f"Converting {len(annotation_files)} annotation files from {dataset_dir.name}...")
    
    converted_count = 0
    error_count = 0
    
    for annotation_file in tqdm(annotation_files, desc=f'Converting {dataset_dir.name}'):
        try:
            # Find corresponding image
            img_name = annotation_file.stem + '.jpg'
            img_path = images_dir / img_name
            
            if not img_path.exists():
                print(f"Warning: Image {img_path} not found for annotation {annotation_file}")
                error_count += 1
                continue
            
            # Get image dimensions
            try:
                with Image.open(img_path) as img:
                    img_size = img.size  # (width, height)
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                error_count += 1
                continue
            
            # Convert annotations
            yolo_lines = []
            
            with open(annotation_file, 'r') as f:
                for line_num, line in enumerate(f.readlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse VisDrone annotation
                        # Format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                        parts = line.split(',')
                        if len(parts) < 6:
                            continue
                        
                        x, y, w, h = map(int, parts[:4])
                        score = int(parts[4])
                        object_category = int(parts[5])
                        
                        # Skip ignored regions (class 0) and invalid boxes
                        if object_category == 0 or w <= 0 or h <= 0:
                            continue
                        
                        # Convert to 0-indexed class (VisDrone classes 1-10 -> YOLO classes 0-9)
                        if object_category < 1 or object_category > 10:
                            continue
                        
                        yolo_class = object_category - 1
                        
                        # Convert bounding box to YOLO format
                        yolo_box = convert_visdrone_box_to_yolo(img_size, [x, y, w, h])
                        
                        # Validate normalized coordinates
                        if all(0 <= coord <= 1 for coord in yolo_box):
                            yolo_line = f"{yolo_class} {' '.join(f'{coord:.6f}' for coord in yolo_box)}"
                            yolo_lines.append(yolo_line)
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line {line_num} in {annotation_file}: {line} - {e}")
                        continue
            
            # Save YOLO format annotations
            label_path = labels_dir / annotation_file.name
            with open(label_path, 'w') as f:
                for line in yolo_lines:
                    f.write(line + '\n')
            
            converted_count += 1
            
        except Exception as e:
            print(f"Error processing {annotation_file}: {e}")
            error_count += 1
    
    print(f"✓ Conversion complete for {dataset_dir.name}:")
    print(f"  - Successfully converted: {converted_count} files")
    print(f"  - Errors: {error_count} files")
    print(f"  - Labels saved to: {labels_dir}")

def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(description='Convert VisDrone annotations to YOLO format')
    parser.add_argument('--data_root', type=str, 
                       default='../../data/raw/visdrone',
                       help='Root directory containing VisDrone dataset')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    script_dir = Path(__file__).parent
    data_root = (script_dir / args.data_root).resolve()
    
    if not data_root.exists():
        print(f"Error: Data root directory not found: {data_root}")
        sys.exit(1)
    
    print("=" * 50)
    print("VisDrone to YOLO Annotation Converter")
    print("=" * 50)
    print(f"Data root: {data_root}")
    print()
    
    # Convert all dataset splits
    splits = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']
    
    for split in splits:
        split_dir = data_root / split
        if split_dir.exists():
            convert_visdrone_annotations(split_dir)
            print()
        else:
            print(f"Warning: Split directory not found: {split_dir}")
    
    print("=" * 50)
    print("Conversion completed!")
    print("Ready for YOLO training.")
    print("=" * 50)

if __name__ == "__main__":
    main()