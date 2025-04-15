import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Create a folder to save augmented images
output_dir = "./augmented_images"
os.makedirs(output_dir, exist_ok=True)

# Load cat.png from the current directory
image_path = "./cat.png"
image = Image.open(image_path)
# Define a series of data augmentation methods
transformations = {
    "original": transforms.Compose([]),  # Original image
    "horizontal_flip": transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),
    "vertical_flip": transforms.Compose([transforms.RandomVerticalFlip(p=1)]),
    "rotation_45": transforms.Compose([transforms.RandomRotation(45)]),
    "rotation_90": transforms.Compose([transforms.RandomRotation(90)]),
    "color_jitter": transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)]),
    "grayscale": transforms.Compose([transforms.Grayscale(num_output_channels=3)]),
    "random_crop": transforms.Compose([transforms.RandomResizedCrop(size=(128, 128))]),
    "center_crop": transforms.Compose([transforms.CenterCrop(size=(128, 128))]),
}

# Prepare a grid for visualization
num_transforms = len(transformations)
fig, axes = plt.subplots(1, num_transforms, figsize=(15, 5))

# Apply each augmentation method to the image and visualize
for idx, (name, transform) in enumerate(transformations.items()):
    # Apply data augmentation
    transformed_image = transform(image)
    
    # Save the augmented image
    save_path = os.path.join(output_dir, f"{name}.png")
    transformed_image.save(save_path)
    
    # Visualize in the grid
    axes[idx].imshow(transformed_image)
    axes[idx].set_title(name)
    axes[idx].axis("off")

# Save the combined visualization
plt.tight_layout()
combined_save_path = os.path.join(output_dir, "combined_visualization.png")
plt.savefig(combined_save_path)
plt.show()

print(f"All augmented images and the combined visualization have been saved to {output_dir}")