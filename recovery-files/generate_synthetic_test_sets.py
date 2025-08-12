# generate_synthetic_test_sets.py
import cv2
import numpy as np
from pathlib import Path
import shutil
from weather_augmentation import add_fog, add_rain, add_night

def generate_synthetic_test_sets():
    """Generate all synthetic test sets from clean test images"""
    
    # Paths (Protocol Compliant)
    clean_test = Path('../../data/raw/visdrone/VisDrone2019-DET-test-dev')
    synthetic_base = Path('../../data/synthetic_test')
    
    # Weather configurations for test sets (Protocol Compliant Parameters)
    weather_configs = {
        'VisDrone2019-DET-test-fog': lambda img: add_fog(img, density=0.5),
        'VisDrone2019-DET-test-rain': lambda img: add_rain(img, intensity=0.6), 
        'VisDrone2019-DET-test-night': lambda img: add_night(img, darkness=0.6),
        'VisDrone2019-DET-test-mixed': lambda img: add_mixed_weather(img)
    }
    
    for weather_name, weather_func in weather_configs.items():
        print(f"Generating {weather_name}...")
        
        # Create output directory
        out_dir = synthetic_base / weather_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy structure
        for subdir in ['images', 'labels']:
            (out_dir / subdir).mkdir(exist_ok=True)
        
        # Process images
        img_dir = clean_test / 'images'
        for img_path in img_dir.glob('*.jpg'):
            # Read image
            img = cv2.imread(str(img_path))
            
            # Apply weather effect
            augmented = weather_func(img)
            
            # Save augmented image
            cv2.imwrite(str(out_dir / 'images' / img_path.name), augmented)
            
            # Copy corresponding label
            label_path = clean_test / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, out_dir / 'labels' / label_path.name)
        
        print(f"✓ Generated {weather_name}: {len(list((out_dir / 'images').glob('*.jpg')))} images")

def add_mixed_weather(image):
    """Apply protocol-compliant mixed weather effects (Trial 5 specification)"""
    # Protocol: Combined weather probability 0.2, specific combinations
    
    # Select combination based on protocol Trial 5 maximum specifications
    combinations = [
        ('fog', 'rain'),    # Most common combination
        ('fog', 'night'),   # Low visibility + darkness
        ('rain', 'night')   # Precipitation + darkness  
    ]
    
    selected = np.random.choice(len(combinations))
    effects = combinations[selected]
    
    # Apply effects with protocol-compliant parameters
    if 'fog' in effects:
        image = add_fog(image, density=0.4)  # Reduced for combination
    if 'rain' in effects:
        image = add_rain(image, intensity=0.5)  # Reduced for combination
    if 'night' in effects:
        image = add_night(image, darkness=0.5)  # Reduced for combination
    
    return image

if __name__ == "__main__":
    generate_synthetic_test_sets()