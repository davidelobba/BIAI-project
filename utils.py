import numpy as np
import torch
from tqdm import tqdm
import yaml
from torchvision import datasets, transforms
from PIL import Image


def upsample_numpy_image(image_np, target_size=(224, 224)):
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8).transpose(1, 2, 0))
    image_pil_upsampled = image_pil.resize(target_size, Image.BILINEAR)
    return np.array(image_pil_upsampled).transpose(2, 0, 1) / 255.0

def get_classification_and_confidence(individual, model, dataset, transform=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if dataset == 'mnist':
        image = torch.reshape(torch.tensor(individual, dtype=torch.float32), (1, 1, 224, 224)).to(device)
    else:
        image = torch.reshape(torch.tensor(individual, dtype=torch.float32), (1, 3, 224, 224)).to(device)

    if transform is not None:
        image = transform[dataset](image)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()]

    return predicted.item(), confidence.item()

def get_classification_and_confidence_cppn(individual, model, dataset, transform):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        if dataset == 'mnist':
            image = torch.reshape(individual, (1, 1, 224, 224))
        else:
            image = torch.reshape(individual, (1, 3, 224, 224))
        if transform is not None:
            image = transform[dataset](image)
            
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()]


    return predicted.item(), confidence.item()

def test_model(model, criterion, dataloader, device):
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)


    print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

def dataset_loader():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=data_transforms['train'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=data_transforms['val'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    dataloader = {'train': trainloader, 'val': testloader}
    
    return dataloader

def load_config(file_path="config.yaml"):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def upsample_numpy_image(image_np, dataset, target_size=(224, 224)):
    if dataset == 'mnist':
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8).squeeze())
    else:
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8).transpose(1, 2, 0))
    image_pil_upsampled = image_pil.resize(target_size, Image.BILINEAR)

    if dataset == 'mnist':
        return np.array(image_pil_upsampled) / 255.0
    else:
        return np.array(image_pil_upsampled).transpose(2, 0, 1) / 255.0

def get_transform():
    transform = {
        'mnist': transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'cifar10': transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'imagenet': transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return transform