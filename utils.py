import numpy as np
import torch
from tqdm import tqdm
import yaml
from torchvision import datasets, transforms
from PIL import Image




def get_classification_and_confidence(individual, model):
    # Create the image from the individual
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(np.array(individual).reshape((3, 224, 224))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # Get the model's predictions
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate the confidence of the prediction
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()]

    return predicted.item(), confidence.item()

def get_classification_and_confidence_cppn(individual, model):
    # Create the image from the individual
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        image = torch.tensor(np.array(individual).reshape((3, 224, 224))).float().unsqueeze(0).to(device)
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

def upsample_numpy_image(image_np, target_size=(224, 224)):
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8).transpose(1, 2, 0))
    image_pil_upsampled = image_pil.resize(target_size, Image.BILINEAR)
    return np.array(image_pil_upsampled).transpose(2, 0, 1) / 255.0