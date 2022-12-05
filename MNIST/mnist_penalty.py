import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix


def load_train_data(batch_size):
    transform_train = transforms.Compose([
        torchvision.transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(param_mean, param_std),
    ])
    
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    
    return train_loader

def load_test_data(batch_size):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(param_mean, param_std),
    ])
    
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = num_workers)
    
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
    confusion = (confusion.T/confusion.sum(axis=1)).T
    confusion = confusion.fill_diagonal_(0)
    print('Acc: %.2f%%' % (100. * Total_acc))
    
    return confusion


def penalty_model_train(model, criterion, optimizer, scheduler):
    
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        correct , sum_loss, total= 0, 0, 0

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
            if i % 200 == 199:
                print('[epoch:%d, iter:%d] Loss: %.05f | Acc: %.2f%%'
                      % (epoch + 1, (i + 1), sum_loss/(i+1), 100. * correct / total)
                      )

        confusion = validation_acc( model, val_loader )
        print("Cost: ", (confusion * cost).sum().item())
        critical_list = np.unravel_index(np.argsort(cost-cost.T, axis=None)[-5:][::-1], cost.shape)
        for no_pair in range(len(critical_list[0])):
            print("Top {} Error Rate: {:.2f}%"
                  .format(no_pair + 1, confusion[critical_list][no_pair]*100))
                
    return model

def model_setup():
    model = torch.load("mnist_baseline").to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.33)
    return model, optimizer, scheduler

# TRAIN
if __name__ == "__main__":
    # Define hyperparameters
    EPOCH = 60
    batch_size = 32
    num_workers = 0
    LR = 1e-4
    num_classes = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pytorch will be using device", device)

    param_mean =  (0.1307)
    param_std = (0.3081)
    
    train_loader = load_train_data(batch_size)
    val_loader = load_test_data(batch_size)

    criterion = nn.CrossEntropyLoss()
    
    cost = torch.load("mnist_cost")

    model, optimizer, scheduler = model_setup()
    penalty_model_train(model, criterion, optimizer, scheduler)
    
    print("Evaluation Finished")
