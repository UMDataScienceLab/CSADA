import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import torchvision.transforms.functional as TF
import random

class RightAngleRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def get_mean_and_std(dataset_path):
    transform_prep = transforms.Compose([
        transforms.ToTensor(),
        ])
    
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform= transform_prep)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in train_dataloader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def load_train_data(path):
    transform_train = transforms.Compose([
        transforms.CenterCrop(960),
        transforms.RandomResizedCrop(224, scale=(0.8, 1), ratio=(1, 1)),
        RightAngleRotation(angles=[0, 90, -90, 180]),
        transforms.ToTensor(),
        transforms.Normalize(param_mean, param_std),
    ])
    
    
    train_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path, path), transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader

def load_test_data(path):

    transform_test = transforms.Compose([
        # input size: 1284*960
        transforms.CenterCrop(960),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(param_mean, param_std),
    ])
    
    test_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path, path), transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader

def model_train(model, criterion, optimizer, scheduler):
    record = np.zeros(EPOCH)
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            if i % 100 == 99:
                print('[epoch:%d, iter:%d] Loss: %.05f | Acc: %.4f%% '
                      % (epoch + 1, (i + 1), sum_loss / (i + 1), 100. * correct / total))
                
        validation(model, val_loader)
        scheduler.step()
        
    return model, record


def validation(model, val_loader):
    print("Pending Test!")
    model.eval()
    with torch.no_grad():
        confusion = np.zeros((num_classes,num_classes))
        occurrence = np.zeros(num_classes)
        
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, 1)
            confusion += confusion_matrix(labels.cpu(), predicted.cpu(), labels=[*range(num_classes)])
            occurrence += np.bincount(labels.cpu(), minlength=num_classes)

    Total_acc = confusion.trace() / occurrence.sum()
    confusion = confusion / (occurrence.T + 1e-8)

    print('Acc: %.2f%%' % (100. * Total_acc))

    return confusion


# TRAIN
if __name__ == "__main__":
    # Define hyperparameters
    EPOCH = 50
    batch_size = 32
    LR = 1e-4
    num_classes = 20
    
    dataset_path = 'NLM20'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pytorch will be using device", device)
    
    param_mean =  ( 0.4923, 0.4027, 0.3429 )
    param_std = ( 0.1905, 0.2175, 0.2434 )

    
    train_loader = load_train_data("train")
    val_loader = load_test_data("valid")
    test_loader = load_test_data("test")

    
    index_to_DrugName = {value : key for (key, value) in torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train')).class_to_idx.items()}
    DrugName_to_index = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train')).class_to_idx.items()


    model = torch.load("NLM_Pretrain")
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()  
    EPOCH = 10
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=1)
    model, val_record = model_train(model, criterion, optimizer, scheduler)
    
    EPOCH = 20
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
    model, val_record = model_train(model, criterion, optimizer, scheduler)
    
    print("Test Set Accuracy:")
    validation(model, test_loader)