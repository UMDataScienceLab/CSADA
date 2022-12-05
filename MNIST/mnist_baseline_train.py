import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_train_data(batch_size):
    transform_train = transforms.Compose([
        torchvision.transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(param_mean, param_std),
    ])
    
    
    train_set = torchvision.datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    
    return train_loader

def load_test_data(batch_size):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(param_mean, param_std),
    ])
    
    test_set = torchvision.datasets.MNIST(root='./MNIST', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = num_workers)
    
    return test_loader


def model_train(model, criterion, optimizer, scheduler):
    record = np.zeros(EPOCH)
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        correct , sum_loss, total = 0, 0, 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs_pure = model(inputs)
            loss = criterion(outputs_pure, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                print('[epoch:%d, iter:%d] Loss: %.05f | Acc: %.2f%%'
                      % (epoch + 1, (i + 1), sum_loss / (i + 1), 100. * correct / total))
        
        model.eval()
        print("Pending Testing")
        with torch.no_grad():
            correct , total  = 0, 0
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            print('[epoch:%d] Acc: %.2f%%'
                  % (epoch + 1, 100*correct / total))
        
        record[epoch] = correct / total
        scheduler.step()
        
    return model, record    

def create_model():
    model = torchvision.models.resnet34(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

# TRAIN
if __name__ == "__main__":
    # Define hyperparameters
    EPOCH = 300
    batch_size = 32
    num_workers = 6
    LR = 1e-1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pytorch will be using device", device)
    
    param_mean =  (0.1307)
    param_std = (0.3081)
    
    train_loader = load_train_data(batch_size)
    val_loader = load_test_data(batch_size)
    
    model = create_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma=0.33)
    criterion = nn.CrossEntropyLoss()
    model, val_record = model_train(model, criterion, optimizer, scheduler)
    
    print("Training Finished")