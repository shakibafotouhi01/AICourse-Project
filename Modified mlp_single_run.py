import torch

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Enable CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if cuda:
  torch.cuda.empty_cache()
print("Cuda is {0}".format(cuda))

# --- 1. CONFIGURATION ---
# Options: 'CIFAR10' or 'FashionMNIST'
DATASET = ['CIFAR10','FashionMNIST']

# --- 2. SETUP AND PARAMETER DEFINITION ---
BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 3. Loading and Normalizing data (Modified for CIFAR 10)
def dataset_load(dataset_name):
    if dataset_name == 'CIFAR10':
        INPUT_SIZE = 3 * 32 * 32
        CLASSES = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        )
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
    elif dataset_name == 'FashionMNIST':
        INPUT_SIZE = 1 * 28 * 28
        CLASSES = (
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
    else:
        raise ValueError("Invalid dataset name")
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return trainloader, testloader, INPUT_SIZE, len(CLASSES), CLASSES

# --- 4. MODEL DEFINITION (Modified- more depth + batch normalizer)---

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, output_size)
        self.dropout = nn.Dropout(p=0.25)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.activation(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.activation(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.fc6(x)
        return x
    
# --- 5. TRAINING FUNCTION (MODIFIED) ---
def train_model(model, trainloader, criterion, optimizer, scheduler, epochs, DEVICE):
    model.train()
    for epoch in range(epochs):

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr.append(current_lr)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(trainloader):.4f} | LR: {current_lr:.6f}")

    print("Training finished.")
    return lr

# --- 6. EVALUATION FUNCTION ---
def evaluate_model(model, testloader, classes):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate class-wise accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    print(f"\nOverall Accuracy: {100 * correct / total:.2f}%")
    
    print("\nClass-wise Accuracy:")
    class_accuracies = {}
    for i in range(len(classes)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            class_name = classes[i]
            class_accuracies[class_name] = accuracy
            print(f'Accuracy of {class_name:13s}: {accuracy:.2f} %')

    if class_accuracies:
        most_difficult_class = min(class_accuracies, key=class_accuracies.get)
        print(f'\n The **most difficult class** to classify is: **{most_difficult_class}** (Accuracy: {class_accuracies[most_difficult_class]:.2f} %)')

# --- 7. Learning Rate Plotting ---

def plot_learning_curves(lr):
    epochs = range(1, len(lr[:80]) + 1)
    plt.figure(figsize=(12, 5))
        
    plt.plot(epochs, lr[:80], 'g', label='Learning Rate')
    plt.title(f'Learning Rate Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
        #
    
    plt.tight_layout()
    plt.show()
    plt.savefig('pic.png')

# --- 8. EXECUTION ---

if __name__ == "__main__":
    trainloader, testloader, INPUT_SIZE, OUTPUT_SIZE, CLASSES = dataset_load(DATASET[0])
    model = MLP(INPUT_SIZE, OUTPUT_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
    lr = train_model(model, trainloader, criterion, optimizer, scheduler, EPOCHS)
    evaluate_model(model, testloader, CLASSES) 
    plot_learning_curves(lr)