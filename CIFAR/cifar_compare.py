import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def load_train_data(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(param_mean, param_std),
    ])
    
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    
    return train_loader

def load_test_data(batch_size):
    transform_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(param_mean, param_std),
    ])
    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = num_workers)
    
    return test_loader


def NLL_Loss(predicts, labels):
    batch_size = predicts.size()[0]           
    corrects = F.log_softmax(predicts, dim=1)  
    corrects = corrects[range(batch_size), labels]
    prediction_loss = - torch.mean(corrects)
    return prediction_loss

def PairwiseNLL_Loss(predicts, labels, pairs):
    penalty_term =  - torch.log(torch.ones_like(predicts).to(device) - F.softmax(predicts, dim=1) + 1e-8)
    penalty = 0
    for a,b in pairs:
        penalty += penalty_term[labels == a][:, b].sum()/((labels == a).sum() + 1e-8)
        
    return penalty/len(pairs)



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
    print(confusion[a,b])

    return confusion[a,b]


def Adv_Aug_Rej(model, X, y, a, b ):
    X_adv = X.clone().detach()
    X_copy = X_adv.clone().detach()
    y = y.clone().detach()
    eps = 1
    Iters = 5
    step_size = 1e-3
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
    
            if (labels == a).count_nonzero() > 0:
                inputs_adv, success = Adv_Aug_Rej(model, inputs, labels, a, b)
                total_success += success
                attacked += (labels == a).count_nonzero()
                adv_loss = criterion(model(inputs_adv), torch.tensor([a]*inputs_adv.shape[0]).to(device).long())
            else: 
                adv_loss=0
            
            
            lmd = 10
            combined_loss = loss + lmd * adv_loss
            combined_loss.backward()
            optimizer.step()

            # #print loss and acc
            sum_loss += combined_loss.item()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

        pair_error = validation(model, val_loader)     
        
    return model, pair_error


def bias_model_train(model, criterion, optimizer, scheduler):
    
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        correct , sum_loss, total= 0, 0, 0

        for i, data in enumerate(train_loader, 0):
            # prepare data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward + backward
            outputs = model(inputs)

            loss = 0* NLL_Loss(outputs, labels) + PairwiseNLL_Loss(outputs, labels, pairs)
            loss.backward()
            optimizer.step()

            #print loss and acc
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            sum_loss += loss.detach()
            
        pair_error = validation(model, val_loader)
                
    return model, pair_error

def model_setup():
    model = torch.load("cifar_baseline").to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=1)
    return model, optimizer, scheduler

# TRAIN
if __name__ == "__main__":
    # Define hyperparameters
    EPOCH = 10
    batch_size = 256
    num_workers = 0
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    LR = 1e-4
    num_classes = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pytorch will be using device", device)

    param_mean = (0.4914, 0.4822, 0.4465)
    param_std = (0.2471, 0.2435, 0.2616)
    
    train_loader = load_train_data(batch_size)
    val_loader = load_test_data(batch_size)
    model = torch.load("cifar_baseline").to(device)

    criterion = nn.CrossEntropyLoss()
    Global_Record = dict()
    
    for a in range(10):
        for b in range(10):
            if a==b:
                continue
            else:
                pairs = [(a,b)]
            
            print()
            print("Current Pair:", a, "to", b)
            
            model, optimizer, scheduler = model_setup()
            current_record = dict()
            
            current_record["Baeline"] = validation( model, val_loader )
            _, current_record["Adv"] = adv_model_train(model, criterion, optimizer, scheduler)
            
            model, optimizer, scheduler = model_setup()
            _, current_record["Penalty"] = bias_model_train(model, criterion, optimizer, scheduler) 
                        
            Global_Record[(a,b)] = current_record.copy()
            
            torch.save(Global_Record, "Compare_CIFAR")
    
    print("Evaluation Finished")
