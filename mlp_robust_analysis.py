
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
# --- 1. CONFIGURATION: CHANGE THIS VARIABLE ---
# Options: 'CIFAR10' or 'FashionMNIST'
DATASET = ['CIFAR10','FashionMNIST']

# --- 2. SETUP AND PARAMETER DEFINITION ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
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

# --- Evaluate Test Metrics (Loss & Acc) for training---
def evaluate_model_metrics(net, testloader, criterion, device):
    correct = 0
    total = 0
    total_loss = 0.0

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            # Accumulate loss: item() gets the scalar, * labels.size(0) scales it by batch size
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = total_loss / total
    overall_accuracy = 100 * correct / total
    return avg_test_loss, overall_accuracy


# --- 6. TRAINING FUNCTION (MODIFIED for History) ---
def train_model(net, trainloader, criterion, optimizer, device, epochs, testloader):
    print(f"\n Starting training for {DATASET_NAME} on {device} for {epochs} epochs...")

    # Lists to store metrics per epoch
    train_loss_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in range(epochs):
        net.train() # Set model to training mode
        running_loss = 0.0

        # Training Phase
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate Training Loss
        avg_train_loss = running_loss / len(trainloader)
        train_loss_history.append(avg_train_loss)

        # Evaluation Phase after each epoch
        avg_test_loss, test_acc = evaluate_model_metrics(net, testloader, criterion, device)
        test_loss_history.append(avg_test_loss)
        test_acc_history.append(test_acc)

        print(f'Epoch {epoch + 1:2d}: Train Loss: {avg_train_loss:.3f}, Test Loss: {avg_test_loss:.3f}, Test Acc: {test_acc:.2f}%')

    print(' Finished Training.')
    return train_loss_history, test_loss_history, test_acc_history

# --- PLOTTING FUNCTIONS ---
def plot_learning_curves(train_loss, test_loss, test_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: Loss vs. Epochs
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r', label='Training Loss')
    plt.plot(epochs, test_loss, 'b', label='Test Loss')
    plt.title(f'Loss Curves for {DATASET_NAME}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    #

    # Plot 2: Accuracy vs. Epochs
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_acc, 'g', label='Test Accuracy')
    plt.title(f'Test Accuracy Curve for {DATASET_NAME}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    #

    plt.tight_layout()
    plt.show()


# Plot 3: Confusion Matrix
def plot_confusion_matrix(cm, classes):
    """Plots the confusion matrix using Seaborn."""
    # Normalize the confusion matrix by the number of instances per class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, linewidths=.5)

    plt.title(f'Confusion Matrix (Normalized) - {DATASET_NAME}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()



def final_evaluation_and_confusion_matrix(net, testloader, classes, device):
    """Performs final evaluation, calculates CM, and plots the results."""
    all_labels = []
    all_predictions = []

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate overall accuracy
    total = len(all_labels)
    correct = np.sum(np.array(all_labels) == np.array(all_predictions))
    overall_accuracy = 100 * correct / total

    print(f"\n--- FINAL RESULTS for {DATASET_NAME} ---")
    print(f'Overall Test Accuracy: {overall_accuracy:.2f} %')

    # Generate the Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Calculate and print class-wise accuracy
    print("\nClass-wise Accuracy:")
    class_accuracies = {}
    for i in range(len(classes)):
        class_name = classes[i]
        class_correct = cm[i, i]
        class_total = np.sum(cm[i, :])
        accuracy = 100 * class_correct / class_total
        class_accuracies[class_name] = accuracy
        print(f'Accuracy of {class_name:13s}: {accuracy:.2f} %')

    most_difficult_class = min(class_accuracies, key=class_accuracies.get)
    print(f'\n the **most difficult class** to classify is: **{most_difficult_class}** (Accuracy: {class_accuracies[most_difficult_class]:.2f} %)')

    # Plot the matrix
    plot_confusion_matrix(cm, classes)
    return overall_accuracy , class_accuracies


# --- 7. EXECUTION (CORRECTED) ---
if __name__ == '__main__':
    # Set the dataset to run
    DATASET_NAME = DATASET[1] # Example: FashionMNIST

    # Initialize data loaders and classes once
    trainloader, testloader, INPUT_SIZE, OUTPUT_SIZE, CLASSES = dataset_load(DATASET_NAME)

    # Initialize lists/dictionaries to store results across all runs
    num_runs = 10
    all_overall_accuracies = []
    all_class_accuracies = {cls: [] for cls in CLASSES} 
    
    # Variables to store history from the last run for plotting
    representative_train_loss, representative_test_loss, representative_test_acc = None, None, None

    # --- START OF 10-RUN LOOP ---
    for run in range(num_runs):
        print(f"\n==================== STARTING RUN {run + 1}/{num_runs} ====================")

        # 1. Re-initialize Model (Crucial for robust analysis)
        net = MLP(INPUT_SIZE, OUTPUT_SIZE).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

        # 2. Run Training and capture history
        train_loss, test_loss, test_acc = train_model(net, trainloader, criterion, optimizer, DEVICE, EPOCHS, testloader)

        # Store history from this run for later plotting
        representative_train_loss, representative_test_loss, representative_test_acc = train_loss, test_loss, test_acc

        # 3. Run Final Evaluation, Plot CM for THIS run, and Collect Data
        # The CM will be plotted 10 times, once for each unique model initialization
        final_acc, class_accs = final_evaluation_and_confusion_matrix(net, testloader, CLASSES, DEVICE)

        # Store the results for the final summary calculation
        all_overall_accuracies.append(final_acc)
        for cls, acc in class_accs.items():
            all_class_accuracies[cls].append(acc)

    # --- END OF 10-RUN LOOP ---
    
    # ==================== PLOTS & FINAL SUMMARY (Executed ONCE) ====================

    # Plot 1: Learning Curves (using the history from the last run)
    if representative_train_loss:
        print("\n==================== GENERATING REPRESENTATIVE LEARNING CURVES ====================")
        plot_learning_curves(representative_train_loss, representative_test_loss, representative_test_acc)
    
    # --- FINAL SUMMARY AFTER ALL RUNS ---
    print("\n==================== FINAL SUMMARY OVER 10 RUNS ====================")

    all_overall_accuracies = np.array(all_overall_accuracies)
    mean_acc = np.mean(all_overall_accuracies)
    # ddof=1 for sample standard deviation (unbiased estimate)
    std_acc  = np.std(all_overall_accuracies, ddof=1) 

    print(f"\nOverall Accuracy Across {num_runs} Runs: {mean_acc:.2f}% ± {std_acc:.2f}%")

    # ---- Mean class accuracies across runs ----
    mean_class_accuracies = {}
    for cls, acc_list in all_class_accuracies.items():
        acc_arr = np.array(acc_list)
        mean_class_accuracies[cls] = np.mean(acc_arr)

    print("\nMean Class Accuracies Across 10 Runs (Sorted by Difficulty):")
    # Sort the mean class accuracies to clearly identify the most difficult classes
    sorted_mean_acc = sorted(mean_class_accuracies.items(), key=lambda item: item[1])
    for cls, acc in sorted_mean_acc:
        print(f"  {cls:12s}: {acc:.2f}%")

    # ---- Most difficult class across runs ----
    most_difficult = min(mean_class_accuracies, key=mean_class_accuracies.get)
    print(f"\n Most Difficult Class Across All Runs: **{most_difficult}** "

          f"(Mean Accuracy: {mean_class_accuracies[most_difficult]:.2f}%)")
