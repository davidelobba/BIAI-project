import torch


from tqdm import tqdm
from dataset import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def evaluate_resnet18(weights_path):
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CIFAR10 test dataset
    dataset = Dataset('cifar10')

    # Load the ResNet18 model and its weights
    network = models.resnet18(pretrained=True)
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, 10)
    network = network.to(device)

    # Perform inference on the test dataset and compute the accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataset.dataset, desc="Evaluating resnet on cifar10", leave=False):
            labels = torch.tensor(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
