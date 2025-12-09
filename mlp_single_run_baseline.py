import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np # Used for easier handling of lists


# Enable CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if cuda:
  torch.cuda.empty_cache()
print("Cuda is {0}".format(cuda))
# --- 1. CONFIGURATION: CHANGE THIS VARIABLE ---
# Options: 'CIFAR10' or 'FashionMNIST'
DATASET = ['CIFAR10','FashionMNIST']

# --- 2. SETUP AND PARAMETER DEFINITION ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 3. Loading and Normalizing data

def dataset_load(DATASET_NAME):
  if DATASET_NAME == 'CIFAR10':
      # CIFAR-10: 32x32 color (3 channels)
      INPUT_SIZE = 3 * 32 * 32
      CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
      # Normalization for 3 channels (R, G, B)
      data_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
      trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
      testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)

  elif DATASET_NAME == 'FashionMNIST':
      # Fashion-MNIST: 28x28 grayscale (1 channel)
      INPUT_SIZE = 1 * 28 * 28
      CLASSES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
      # Normalization for 1 channel (Grayscale)
      data_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,))
      ])
      trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=data_transform)
      testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=data_transform)

  else:
      raise ValueError("Invalid DATASET_NAME. Use 'CIFAR10' or 'FashionMNIST'.")

  # Create DataLoaders
  trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
  testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
  OUTPUT_SIZE = len(CLASSES)
  return trainloader, testloader, INPUT_SIZE ,OUTPUT_SIZE, CLASSES
# --- 4. MODEL DEFINITION ---
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()

        # Consistent hidden layer sizes for both datasets
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        # Flatten the image tensor (adapts to 784 or 3072 input)
        x = self.flatten(x)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 5. TRAINING FUNCTION ---
def train_model(net, trainloader, criterion, optimizer, device, epochs):
    print(f"\n Starting training for {DATASET_NAME} on {device} for {epochs} epochs...")
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
    print('Finished Training.')
# --- 6. EVALUATION FUNCTION ---
def evaluate_model(net, testloader, classes, device):
    correct = 0
    total = 0
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate class-wise accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    overall_accuracy = 100 * correct / total
    print(f'\n--- RESULTS for {DATASET_NAME} ---')
    print(f'Overall Accuracy of the network: {overall_accuracy:.2f} %')

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
        print(f'\n🔥 The **most difficult class** to classify is: **{most_difficult_class}** (Accuracy: {class_accuracies[most_difficult_class]:.2f} %)')


# --- 7. EXECUTION ---
if __name__ == '__main__':
    DATASET_NAME=DATASET[1]
    trainloader, testloader, INPUT_SIZE ,OUTPUT_SIZE, CLASSES = dataset_load(DATASET_NAME)
    net = MLP(INPUT_SIZE, OUTPUT_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    train_model(net, trainloader, criterion, optimizer, DEVICE, EPOCHS)
    evaluate_model(net, testloader, CLASSES, DEVICE)