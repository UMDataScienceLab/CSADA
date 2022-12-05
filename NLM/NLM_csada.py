import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import torchvision.transforms.functional as TF
import random
import os
from torchmetrics import ConfusionMatrix


class RightAngleRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def load_train_data(path):
    transform_train = transforms.Compose([
        # input size: 1284*960
        transforms.CenterCrop(960),
        transforms.RandomResizedCrop(224, scale=(0.8, 1), ratio=(1, 1)),
        RightAngleRotation(angles=[0, 90, -90, 180]),
        transforms.ToTensor(),
        transforms.Normalize(param_mean, param_std),
    ])
    
    
    train_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path, path), transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    
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
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
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
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, 1)
            confusion += compute_confusion(predicted.cpu(), labels.cpu())
    
    Total_acc = confusion.trace() / confusion.sum()
    print("Test Cost: ", (confusion * cost).sum().item())
    confusion = (confusion.T/confusion.sum(axis=1)).T
    confusion = confusion.fill_diagonal_(0)
    print('Acc: %.2f%%' % (100. * Total_acc))
    
    return confusion


def Adv_Aug_Rej(model, X, y, a, b ):
    X_adv = X.clone().detach()
    X_copy = X_adv.clone().detach()
    y = y.clone().detach()
    eps = 1
    Iters = 50
    step_size = 1e-4
    success = 0
    model.eval()

    output = model(X_adv).to(device)
    predicted = torch.argmax(output.data, 1)
    a_index = torch.logical_and((predicted == a), (y == a)).clone()
    for i in range(Iters):

        if a_index.sum() == 0:
            output = model(X_adv[y == a]).to(device)
            predicted = torch.argmax(output.data, 1)
            success = (predicted == b).sum()
            return X_adv[y == a], success
        
        X_alt = X_adv[a_index].clone().detach()
        X_alt.requires_grad = True
        output = model(X_alt).to(device)
        log_prob = F.log_softmax(output, 1)[:, b]
        X_grad = torch.autograd.grad(log_prob, X_alt, grad_outputs=torch.ones_like(log_prob))[0]
        
        with torch.no_grad():
            X_alt = (X_alt + X_grad * step_size).clone().detach()
            X_alt = (X_copy[a_index] + (X_alt - X_copy[a_index]).clamp(-eps, eps)).clone()
            output = model(X_alt).to(device)
            predicted = torch.argmax(output.data, 1)
            updates_index = (predicted == a) + (predicted == b)
            X_adv[chain_index(a_index, updates_index)] = X_alt[updates_index].clone().detach()
            a_index = chain_index(a_index, (predicted == a))

    output = model(X_adv[y == a]).to(device)
    predicted = torch.argmax(output.data, 1)
    success = (predicted == b).sum()

    return X_adv[y == a], success

def chain_index(a_index, updates_index):
    a_index_new = [False]*len(a_index)
    num_index = torch.tensor([*range(len(a_index))])
    num_index[a_index][updates_index]
    for i in num_index[a_index][updates_index]:
        a_index_new[i] = True
    return torch.tensor(a_index_new)

def sample_pair(cost):
    index = np.random.choice(range(num_classes**2), p=(attack_prob / attack_prob.sum()).flatten())
    b = index % num_classes
    a = int((index-b)/num_classes)
    return (a,b)

def adv_model_train(model, criterion, optimizer, attack_prob):
    
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        correct , sum_loss, total, total_success, attacked = 0, 0, 0, 0, 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs_pure = model(inputs)
            loss = criterion(outputs_pure, labels)
    
            a, b = sample_pair(attack_prob)

            
            if (labels == a).count_nonzero() > 0:
                inputs_adv, success = Adv_Aug_Rej(model, inputs, labels, a, b)
                total_success += success
                attacked += (labels == a).count_nonzero()
                adv_loss = criterion(model(inputs_adv), torch.tensor([a]*inputs_adv.shape[0]).to(device).long())
            else: 
                adv_loss=0
            
            
            lmd = 2
            combined_loss = loss + lmd * adv_loss
            combined_loss.backward()
            optimizer.step()

            sum_loss += combined_loss.item()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 50 == 49:
                print('[epoch:%d, iter:%d] Loss: %.05f | Acc: %.2f%% | succ:%.2f%%'
                      % (epoch + 1, (i + 1), sum_loss/(i+1), 100. * correct / total, 100*(total_success+1e-8)/(attacked+1e-8))
                      )

        
        confusion = validation_acc( model, val_loader )
        print("Valid Cost: ", (confusion * cost).sum().item())
        print(confusion[critical_list])
        
        confusion = validation_acc( model, test_loader )
        print("Valid Cost: ", (confusion * cost).sum().item())
        
        print(confusion[critical_list])

        
        
    return model

def remove_diag(matrix):
    res = []
    for idx, ele in enumerate(matrix):
        res.extend([el for idxx, el in enumerate(ele) if idxx != idx])
    return res


def power_normalize(cost, power = 1):
    power_cost = cost**power
    return power_cost/power_cost.sum()

# TRAIN
if __name__ == "__main__":
    # Define hyperparameters
    EPOCH = 10
    batch_size = 32
    num_workers = 0
    dataset_path = 'NLM20'
    LR = 1e-7
    model_name = "NLM20_Baseline"
    
    num_classes = 20
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pytorch will be using device", device)

    param_mean =  np.array(( 0.4923, 0.4027, 0.3429 ))
    param_std = np.array(( 0.1905, 0.2175, 0.2434 ))
    inverse_transform = transforms.Compose([
        transforms.Normalize( (0,0,0), 1/param_std),
        transforms.Normalize( -param_mean, (1,1,1)),
        transforms.ToPILImage()
        ])
    train_loader = load_train_data("train")
    val_loader = load_test_data("valid")
    test_loader = load_test_data("test")
    index_to_DrugName = {value : key for (key, value) in torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train')).class_to_idx.items()}
    DrugName_to_index = {key : value for (key, value) in torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train')).class_to_idx.items()}

    model = torch.load(model_name).to(device)
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    compute_confusion = ConfusionMatrix(num_classes=num_classes)
    cost = np.ones((num_classes,num_classes))
    np.fill_diagonal(cost, 0)
    cost[DrugName_to_index["50111-0434"], DrugName_to_index["00591-0461"]] = 10
    cost[DrugName_to_index["53489-0156"], DrugName_to_index["68382-0227"]] = 10
    cost[DrugName_to_index["68382-0227"], DrugName_to_index["53489-0156"]] = 8
    cost[DrugName_to_index["53746-0544"], DrugName_to_index["00378-0208"]] = 10
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=1)

    attack_prob = power_normalize(cost, power = 3)

    confusion = validation_acc( model, val_loader ) 
    print("Valid Cost: ", (confusion * cost).sum())
    
    top = 4
    critical_list = np.unravel_index(np.argsort(cost, axis=None)[-top:][::-1], cost.shape)
    for no_pair in range(len(critical_list[0])):
        print("Top {} Error Rate: {:.2f}%, Assigned Probability: {:.2f}%"
              .format(no_pair + 1, 
                      confusion[critical_list][no_pair]*100,
                      attack_prob[critical_list][no_pair]*100)) 
    
    confusion_test = validation_acc( model, test_loader ) 
    
    for no_pair in range(len(critical_list[0])):
        print("Top {} Error Rate: {:.2f}%"
              .format(no_pair + 1, confusion_test[critical_list][no_pair]*100)) 
    
    start = time.time()
    adv_model_train(model, criterion, optimizer, attack_prob)
    print("With Rejection Training time:", time.time() - start)
    
    confusion_test = validation_acc( model, test_loader )
    
    for no_pair in range(len(critical_list[0])):
        print("Top {} Error Rate: {:.2f}%"
              .format(no_pair + 1, confusion_test[critical_list][no_pair]*100)) 
    
    print("Evaluation Finished")
