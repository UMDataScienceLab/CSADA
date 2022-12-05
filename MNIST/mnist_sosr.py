import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
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


def validation_acc(model, val_loader):
    print("\nPending Test!")
    compute_confusion = ConfusionMatrix(num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        confusion = torch.zeros((num_classes,num_classes))
        
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = layer_SOSR(model(inputs))
            predicted = torch.argmin(outputs.data, 1)
            confusion += compute_confusion(predicted.cpu(), labels.cpu())

    Total_acc = confusion.trace() / confusion.sum()
    confusion = (confusion.T/confusion.sum(axis=1)).T
    confusion = confusion.fill_diagonal_(0)
    print('Acc: %.2f%%' % (100. * Total_acc))
    
    return confusion

def sosr_loss(regression_output, cost, labels):
    class_cost = torch.tensor(cost).to(device)[labels]
    z = -torch.ones_like(regression_output).to(device)
    z[range(z.shape[0]),labels] = 1
    return torch.log(1+torch.exp((regression_output - class_cost)*z)).sum()/batch_size


def sosr_train(model, criterion, optimizer, scheduler, layer_SOSR):
    
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss= 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            regression_output = layer_SOSR(outputs)
            
            loss = sosr_loss(regression_output, cost, labels)
            loss.backward()
            optimizer.step()
            
            outputs = model(inputs)
            sum_loss += loss.detach()
            if i % 200 == 199:
                print('[epoch:%d, iter:%d] Loss: %.05f' % (epoch + 1, (i + 1), sum_loss/(i+1)))

        confusion = validation_acc( model, val_loader )
        print("Cost: ", (confusion * cost).sum().item())
        critical_list = np.unravel_index(np.argsort(cost-cost.T, axis=None)[-5:][::-1], cost.shape)
        for no_pair in range(len(critical_list[0])):
            print("Top {} Error Rate: {:.2f}%"
                  .format(no_pair + 1, confusion[critical_list][no_pair]*100))
                
    return model, layer_SOSR

def model_setup():
    model = torch.load("mnist_baseline").to(device)
    layer_SOSR = nn.Linear(num_classes, num_classes).to(device)
    optimizer = torch.optim.SGD([*model.parameters(),*layer_SOSR.parameters()], lr=LR, weight_decay=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.33)
    return model, optimizer, scheduler, layer_SOSR

# TRAIN
if __name__ == "__main__":
    # Define hyperparameters
    EPOCH = 60
    batch_size = 32
    num_workers = 6
    LR = 1e-4
    num_classes = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pytorch will be using device", device)

    param_mean =  (0.1307)
    param_std = (0.3081)
    
    train_loader = load_train_data(batch_size)
    val_loader = load_test_data(batch_size)
    model, optimizer, scheduler, layer_SOSR = model_setup()
    
    criterion = nn.CrossEntropyLoss()
    
    cost = torch.load("mnist_cost")
    confusion = validation_acc(model, val_loader)
    print("Cost: ", (confusion * cost).sum().item())
    
    model, optimizer, scheduler, layer_SOSR = model_setup()
    model, layer_SOSR = sosr_train(model, criterion, optimizer, scheduler, layer_SOSR)
    
    print("Evaluation Finished")
