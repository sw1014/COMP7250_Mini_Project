import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import netron
import os


augmentations = {
    "original": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "horizontal_flip_rotation_90": transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "vertical_flip": transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "rotation_45": transforms.Compose([transforms.RandomRotation(45), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "rotation_90": transforms.Compose([transforms.RandomRotation(90), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "color_jitter": transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "grayscale": transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "random_crop": transforms.Compose([transforms.RandomResizedCrop(size=(32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "center_crop": transforms.Compose([transforms.CenterCrop(size=(32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "horizontal_flip_rotation_90_color_jitter": transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                            transforms.RandomRotation(90),
                                                            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                                                            transforms.RandomResizedCrop(size=(32, 32)),
                                                            transforms.ToTensor(), 
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}


#defeine a convolutional neural network
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


#save model structure 
# dummy_input = torch.randn(1, 3, 32, 32)
# onnx_path = "network_structure.onnx"
# torch.onnx.export(net, dummy_input, onnx_path, verbose=True)
# print(f"Network structure has been saved as '{onnx_path}'.")
# netron.start(onnx_path)


# Training and evaluation function
def train_and_evaluate(augmentation_name, transform, epochs=50, batch_size=40):
    print(f"Training with augmentation: {augmentation_name}")
    # Load datasets with the specified augmentation
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_losses = []
    test_losses = []
    test_accuracies = []

    # Training loop
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(trainloader))

        # Evaluation loop
        net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(testloader, desc=f"Epoch {epoch + 1}/{epochs} - Testing"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = lossFunction(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_losses.append(test_loss / len(testloader))
        test_accuracies.append(100.0 * correct / total)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.3f}, Test Loss: {test_losses[-1]:.3f}, Test Accuracy: {test_accuracies[-1]:.2f}%")

    # Save model weights
    model_path = f"./checkpoints/{augmentation_name}_model.pth"
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(net.state_dict(), model_path)

    # Save training and testing curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Testing Loss ({augmentation_name})")
    plt.legend()
    plt.savefig(f"./checkpoints/{augmentation_name}_loss_curve.png")
    plt.close()

    return test_accuracies[-1]



#main function to train and evaluate the model with different augmentations
results = {}
for aug_name, aug_transform in augmentations.items():
    accuracy = train_and_evaluate(aug_name, aug_transform, epochs=100, batch_size=40)
    results[aug_name] = accuracy

# Print and save results
print("Final Results:")
for aug_name, accuracy in results.items():
    print(f"{aug_name}: {accuracy:.2f}%")
with open("./checkpoints/results.txt", "w") as f:
    for aug_name, accuracy in results.items():
        f.write(f"{aug_name}: {accuracy:.2f}%\n")