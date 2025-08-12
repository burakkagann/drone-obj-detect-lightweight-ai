"""
Weather Augmentation Module for Drone Object Detection
Protocol-Compliant Implementation for Phase 1 Testing

Implements atmospheric effects specifically designed for drone/aerial imagery
following the specifications in yolo-augmentation-protocol.md
"""

import cv2
import numpy as np
from typing import Tuple


def add_fog(image: np.ndarray, density: float = 0.5) -> np.ndarray:
    """
    Add fog effect to drone image using atmospheric scattering model
    
    Args:
        image: BGR image (numpy array, uint8)
        density: Fog density from 0.0 to 1.0 (protocol range: [0.3, 0.8])
        
    Returns:
        Augmented BGR image (uint8)
    """
    # Validate input parameters according to protocol
    density = np.clip(density, 0.0, 1.0)
    
    # Convert to float for processing
    fog_image = image.astype(np.float32) / 255.0
    h, w = fog_image.shape[:2]
    
    # Create atmospheric scattering model for aerial perspective
    # Distance increases from center (drone viewpoint)
    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # Calculate distance from center (normalized)
    distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    normalized_distance = distance_from_center / max_distance
    
    # Create fog mask with atmospheric scattering
    # More fog at greater distances (typical for aerial imagery)
    fog_intensity = density * (0.3 + 0.7 * normalized_distance)
    fog_intensity = np.clip(fog_intensity, 0, 1)
    
    # Apply fog using atmospheric scattering equation: I = I₀ * e^(-βd) + A(1 - e^(-βd))
    # Where β is scattering coefficient, d is distance, A is atmospheric light
    atmospheric_light = 0.9  # Bright fog color
    transmission = np.exp(-fog_intensity * 2.5)  # Scattering coefficient
    
    # Apply to each channel
    for channel in range(3):
        fog_image[:, :, channel] = (fog_image[:, :, channel] * transmission + 
                                   atmospheric_light * (1 - transmission))
    
    # Convert back to uint8
    fog_image = np.clip(fog_image * 255, 0, 255).astype(np.uint8)
    
    return fog_image


def add_rain(image: np.ndarray, intensity: float = 0.6) -> np.ndarray:
    """
    Add rain effect to drone image with motion blur for aerial perspective
    
    Args:
        image: BGR image (numpy array, uint8)
        intensity: Rain intensity from 0.0 to 1.0 (protocol range: [0.3, 0.7])
        
    Returns:
        Augmented BGR image (uint8)
    """
    # Validate input parameters according to protocol
    intensity = np.clip(intensity, 0.0, 1.0)
    
    rain_image = image.copy()
    h, w = rain_image.shape[:2]
    
    # Generate rain streaks appropriate for drone altitude
    num_drops = int(intensity * 2000)  # Scale based on intensity
    
    # Create rain overlay
    rain_overlay = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(num_drops):
        # Random position
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        
        # Rain streak length varies with intensity
        streak_length = np.random.randint(10, int(30 * intensity))
        
        # Slight angle for drone movement effect (5-15 degrees)
        angle = np.random.uniform(-15, -5)  # Negative for downward rain
        
        # Calculate end point
        end_x = int(x + streak_length * np.sin(np.radians(angle)))
        end_y = int(y + streak_length * np.cos(np.radians(angle)))
        
        # Ensure end point is within image bounds
        end_x = np.clip(end_x, 0, w-1)
        end_y = np.clip(end_y, 0, h-1)
        
        # Draw rain streak with varying intensity
        streak_intensity = np.random.randint(100, 255)
        cv2.line(rain_overlay, (x, y), (end_x, end_y), streak_intensity, 1)
    
    # Apply motion blur to rain streaks
    blur_kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32) / 3
    rain_overlay = cv2.filter2D(rain_overlay, -1, blur_kernel)
    
    # Blend rain with original image
    rain_mask = rain_overlay.astype(np.float32) / 255.0
    rain_color = np.array([200, 200, 255], dtype=np.float32)  # Slightly blue-tinted rain
    
    for channel in range(3):
        rain_image[:, :, channel] = (rain_image[:, :, channel].astype(np.float32) * (1 - rain_mask * intensity) + 
                                   rain_color[channel] * rain_mask * intensity)
    
    # Add atmospheric haze effect for heavy rain
    if intensity > 0.5:
        haze_intensity = (intensity - 0.5) * 0.4  # Mild haze for heavy rain
        rain_image = rain_image.astype(np.float32)
        rain_image = rain_image * (1 - haze_intensity) + 180 * haze_intensity
        rain_image = np.clip(rain_image, 0, 255)
    
    return rain_image.astype(np.uint8)


def add_night(image: np.ndarray, darkness: float = 0.6) -> np.ndarray:
    """
    Add night effect to drone image with realistic low-light simulation
    
    Args:
        image: BGR image (numpy array, uint8)
        darkness: Darkness level from 0.0 to 1.0 (protocol range: [0.3, 0.7])
        
    Returns:
        Augmented BGR image (uint8)
    """
    # Validate input parameters according to protocol
    darkness = np.clip(darkness, 0.0, 1.0)
    
    # Convert to float for processing
    night_image = image.astype(np.float32) / 255.0
    
    # Apply brightness reduction with gamma correction for realistic night effect
    # Lower gamma makes darker areas more visible (simulates eye adaptation)
    gamma = 1.0 + darkness * 0.8  # Gamma range: 1.0 to 1.8
    brightness_reduction = 1.0 - darkness * 0.7  # Maintain some visibility
    
    # Apply gamma correction and brightness reduction
    night_image = np.power(night_image, gamma) * brightness_reduction
    
    # Add noise to simulate low-light sensor noise
    # Poisson noise for low-light conditions
    noise_scale = darkness * 0.15  # Increase noise with darkness
    
    # Generate noise with proper scaling
    noise = np.random.poisson(night_image * 50) / 50.0 - night_image
    noise = noise * noise_scale
    
    # Add Gaussian noise for additional realism
    gaussian_noise = np.random.normal(0, darkness * 0.02, night_image.shape)
    
    # Combine image with noise
    night_image = night_image + noise + gaussian_noise
    
    # Add slight blue tint for moonlight effect
    if darkness > 0.4:
        blue_tint = (darkness - 0.4) * 0.3
        night_image[:, :, 0] *= (1 + blue_tint * 0.1)  # Slight blue increase
        night_image[:, :, 1] *= (1 + blue_tint * 0.05) # Slight green increase
        night_image[:, :, 2] *= (1 + blue_tint * 0.2)  # More blue for moonlight
    
    # Ensure values are in valid range and convert back to uint8
    night_image = np.clip(night_image * 255, 0, 255).astype(np.uint8)
    
    return night_image


def validate_weather_params(fog_density: float = None, rain_intensity: float = None, 
                          night_darkness: float = None) -> Tuple[bool, str]:
    """
    Validate weather parameters against protocol specifications
    
    Args:
        fog_density: Fog density to validate
        rain_intensity: Rain intensity to validate  
        night_darkness: Night darkness to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []
    
    if fog_density is not None:
        if not (0.3 <= fog_density <= 0.8):
            errors.append(f"Fog density {fog_density} outside protocol range [0.3, 0.8]")
    
    if rain_intensity is not None:
        if not (0.3 <= rain_intensity <= 0.7):
            errors.append(f"Rain intensity {rain_intensity} outside protocol range [0.3, 0.7]")
    
    if night_darkness is not None:
        if not (0.3 <= night_darkness <= 0.7):
            errors.append(f"Night darkness {night_darkness} outside protocol range [0.3, 0.7]")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "All parameters within protocol specifications"


# Protocol compliance verification
def get_weather_config():
    """
    Return protocol-compliant weather configuration for Phase 1 testing
    
    Returns:
        Dictionary with default weather parameters matching protocol
    """
    return {
        'fog': {'density': 0.5, 'range': [0.3, 0.8]},
        'rain': {'intensity': 0.6, 'range': [0.3, 0.7]}, 
        'night': {'darkness': 0.6, 'range': [0.3, 0.7]},
        'mixed': {
            'fog_density': 0.4,
            'rain_intensity': 0.5,
            'night_darkness': 0.5
        }
    }


if __name__ == "__main__":
    # Test implementation with sample image
    print("Weather Augmentation Module - Protocol Compliance Test")
    print("=" * 50)
    
    # Test parameter validation
    config = get_weather_config()
    print(f"Default configuration: {config}")
    
    # Validate parameters
    for weather_type, params in config.items():
        if weather_type != 'mixed':
            param_name = list(params.keys())[0]
            param_value = params[param_name]
            valid, msg = validate_weather_params(**{param_name: param_value})
            print(f"{weather_type.capitalize()}: {param_value} - {'✓ Valid' if valid else '✗ Invalid'}")
    
    print("\nModule ready for synthetic test set generation!")