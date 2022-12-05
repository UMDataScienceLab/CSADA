import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
import torchvision.transforms.functional as TF
import os
import random


class RightAngleRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def load_train_data(path):
    transform_train = transforms.Compose([
        transforms.CenterCrop(960),
        transforms.RandomResizedCrop(224, scale=(0.8, 1), ratio=(1, 1)),
        RightAngleRotation(angles=[0, 90, -90, 180]),
        transforms.ToTensor(),
        transforms.Normalize(param_mean, param_std),
    ])
    
    
    train_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path, path), transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
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
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return test_loader


def NLL_Loss(predicts, labels):
    batch_size = predicts.size()[0]           
    corrects = F.log_softmax(predicts, dim=1)  
    corrects = corrects[range(batch_size), labels] 
    prediction_loss = - torch.mean(corrects)
    return prediction_loss


def full_pairwise_nll_loss(outputs, labels, cost):
    neg_log =  - torch.log(torch.ones_like(outputs).to(device) - F.softmax(outputs, dim=1) + 1e-8)
    class_cost = torch.tensor(cost).to(device)[labels]
    return (class_cost*neg_log).sum(axis=1).mean()


def validation_acc(model, val_loader):
    print("\nPending Test!")
    compute_confusion = ConfusionMatrix(num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        confusion = torch.zeros((num_classes,num_classes))
        
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, 1)
            confusion += compute_confusion(predicted.cpu(), labels.cpu())
    
    Total_acc = confusion.trace() / confusion.sum()
    print("Test Cost: ", (confusion * cost).sum().item())
    confusion = (confusion.T/confusion.sum(axis=1)).T
    confusion = confusion.fill_diagonal_(0)
    print('Acc: %.2f%%' % (100. * Total_acc))
    
    return confusion


def bias_model_train(model, criterion, optimizer, scheduler):
    
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        correct , sum_loss, total = 0, 0, 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = NLL_Loss(outputs, labels) + 5 * full_pairwise_nll_loss(outputs, labels, cost)
            loss.backward()
            optimizer.step()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            sum_loss += loss.detach()
            if i % 50 == 49:
                print('[epoch:%d, iter:%d] Loss: %.05f | Acc: %.2f%%'
                      % (epoch + 1, (i + 1), sum_loss/(i+1), 100. * correct / total)
                      )

        confusion = validation_acc(model, val_loader)
        for no_pair in range(len(critical_list[0])):
            print("Top {} Error Rate: {:.2f}%"
                  .format(no_pair + 1, confusion[critical_list][no_pair]*100)) 
                
    return model

def model_setup():
    model = torch.load(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.33)
    return model, optimizer, scheduler

# TRAIN
if __name__ == "__main__":
    # Define hyperparameters
    EPOCH = 50
    batch_size = 32
    num_workers = 0
    dataset_path = 'NLM20'
    LR = 1e-3
    model_name = "NLM_ResNet9"
    
    num_classes = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pytorch will be using device", device)

    param_mean =  np.array(( 0.4923, 0.4027, 0.3429 ))
    param_std = np.array(( 0.1905, 0.2175, 0.2434 ))

    train_loader = load_train_data("train")
    val_loader = load_test_data("valid")
    test_loader = load_test_data("test")
    
    index_to_DrugName = {value : key for (key, value) in torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train')).class_to_idx.items()}
    DrugName_to_index = {key : value for (key, value) in torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train')).class_to_idx.items()}

    criterion = nn.CrossEntropyLoss()  

    compute_confusion = ConfusionMatrix(num_classes=num_classes)
    cost = np.ones((num_classes,num_classes))
    np.fill_diagonal(cost, 0)
    cost[DrugName_to_index["50111-0434"], DrugName_to_index["00591-0461"]] = 10
    cost[DrugName_to_index["53489-0156"], DrugName_to_index["68382-0227"]] = 10
    cost[DrugName_to_index["68382-0227"], DrugName_to_index["53489-0156"]] = 8
    cost[DrugName_to_index["53746-0544"], DrugName_to_index["00378-0208"]] = 10

    top = 4
    critical_list = np.unravel_index(np.argsort(cost, axis=None)[-top:][::-1], cost.shape)
    
    model = torch.load(model_name).to(device)
    confusion_test = validation_acc( model, val_loader ) 
    
    for no_pair in range(len(critical_list[0])):
        print("Top {} Error Rate: {:.2f}%"
              .format(no_pair + 1, confusion_test[critical_list][no_pair]*100)) 
    
       
    model = torch.load(model_name).to(device)
    confusion_test = validation_acc( model, test_loader ) 
    
    for no_pair in range(len(critical_list[0])):
        print("Top {} Error Rate: {:.2f}%"
              .format(no_pair + 1, confusion_test[critical_list][no_pair]*100)) 

    model, optimizer, scheduler = model_setup()
    bias_model_train(model, criterion, optimizer, scheduler)
    
    confusion_test = validation_acc( model, test_loader ) 
    
    for no_pair in range(len(critical_list[0])):
        print("Top {} Error Rate: {:.2f}%"
              .format(no_pair + 1, confusion_test[critical_list][no_pair]*100)) 
    
    
    print("Evaluation Finished")


