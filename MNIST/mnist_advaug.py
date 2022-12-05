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

def validation_acc(model, val_loader):
    print("\nPending Test!")
    model.eval()
    compute_confusion = ConfusionMatrix(num_classes=num_classes)
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
    print('Acc: %.3f%%' % (100. * Total_acc))
    
    return confusion


def Adv_Aug_Rej(model, X, y, a, b ):
    X_adv = X.clone().detach()
    X_copy = X_adv.clone().detach()
    y = y.clone().detach()
    eps = 3
    Iters = 5
    step_size = 0.05
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

def sample_pair(attack_prob):
    index = np.random.choice(range(100), p=(attack_prob / attack_prob.sum()).flatten())
    b = index % 10
    a = int((index-b)/10)
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
            
            combined_loss = loss + lmd * adv_loss
            combined_loss.backward()
            optimizer.step()

            sum_loss += combined_loss.item()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 500 == 499:
                print('[epoch:%d, iter:%d] Loss: %.05f | Acc: %.2f%% | succ:%.2f%%'
                      % (epoch + 1, (i + 1), sum_loss/(i+1), 100. * correct / total, 100*(total_success+1e-8)/(attacked+1e-8))
                      )

        confusion = validation_acc( model, val_loader )
        print("Cost: ", (confusion * cost).sum().item())
        for no_pair in range(len(critical_list[0])):
            print("Top {} Error Rate: {:.2f}%"
                  .format(no_pair + 1, confusion[critical_list][no_pair]*100))

        
    return model

def remove_diag(matrix):
    res = []
    for idx, ele in enumerate(matrix):
        res.extend([el for idxx, el in enumerate(ele) if idxx != idx])
    return res


def power_normalize(cost, power = 1):
    power_cost = cost**power
    return power_cost/power_cost.sum()


def model_setup():
    model = torch.load("mnist_baseline").to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-4, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=1)
    return model, optimizer, scheduler

# TRAIN
if __name__ == "__main__":
    # Define hyperparameters
    EPOCH = 10
    batch_size = 32
    num_workers = 4
    LR = 5e-7
    num_classes = 10
    

    lmd = 5
    print("current_lmd",lmd)

    # Define GPU usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pytorch will be using device", device)

    param_mean =  (0.1307)
    param_std = (0.3081)
    
    train_loader = load_train_data(batch_size)
    val_loader = load_test_data(batch_size)

    criterion = nn.CrossEntropyLoss()  
    model, _  , _ = model_setup()
    
    confusion = validation_acc( model, val_loader )

    cost = torch.load("mnist_cost")
    top = 5
    critical_list = np.unravel_index(np.argsort(cost-cost.T, axis=None)[-top:][::-1], cost.shape)
    
    attack_prob = power_normalize(cost, power = 3)
    
    print("Cost: ", (confusion * cost).sum().item())
    for no_pair in range(len(critical_list[0])):
        print("Top {} Error Rate: {:.2f}%, Assigned Probability: {:.2f}%"
              .format(no_pair + 1, 
                      confusion[critical_list][no_pair]*100,
                      attack_prob[critical_list][no_pair]*100)) 
    
    model, optimizer, scheduler  = model_setup()
    adv_model_train( model, criterion, optimizer, attack_prob )

    print("Evaluation Finished")
