import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

# Define the same network structure as in training
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define data augmentation methods, consistent with training
augmentations = {
    "original": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "horizontal_flip": transforms.Compose([transforms.RandomHorizontalFlip(p=1), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "vertical_flip": transforms.Compose([transforms.RandomVerticalFlip(p=1), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "rotation_45": transforms.Compose([transforms.RandomRotation(45), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "rotation_90": transforms.Compose([transforms.RandomRotation(90), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "color_jitter": transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "grayscale": transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "random_crop": transforms.Compose([transforms.RandomResizedCrop(size=(32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "horizontal_flip_rotation_90_color_jitter": transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                        transforms.RandomRotation(90),
                                                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                                                        transforms.RandomResizedCrop(size=(32, 32)),
                                                        transforms.ToTensor(), 
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

# Function to test the model
def test_model(augmentation_name, model_path, batch_size=40):
    print(f"Testing with augmentation: {augmentation_name}")
    original_transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load the test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=original_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc=f"Testing {augmentation_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Accuracy for {augmentation_name}: {accuracy:.2f}%")
    return accuracy

# Main function
if __name__ == "__main__":
    results = {}
    checkpoints_dir = "./checkpoints"  # Directory where model weights are saved

    # Iterate through all data augmentation methods
    for aug_name in augmentations.keys():
        model_path = os.path.join(checkpoints_dir, f"{aug_name}_model.pth")
        if os.path.exists(model_path):
            accuracy = test_model(aug_name, model_path)
            results[aug_name] = accuracy
        else:
            print(f"Model file for {aug_name} not found: {model_path}")

    # Print and save results
    print("Final Test Results:")
    for aug_name, accuracy in results.items():
        print(f"{aug_name}: {accuracy:.2f}%")
    with open("./checkpoints/test_results.txt", "w") as f:
        for aug_name, accuracy in results.items():
            f.write(f"{aug_name}: {accuracy:.2f}%\n")