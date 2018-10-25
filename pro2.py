# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
#%%
def mod_batch(x):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
#    transform_train = transforms.Compose([
#        transforms.Resize((40,40)),
#        transforms.RandomAffine(de, translate=(tr,tr)),
#        transforms.ToTensor()
#    ])
    
    trainset = ImageFolder('./release/train', transform=transform)
    testset = ImageFolder('./release/val', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=x, 
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=x,
                                             shuffle=True, num_workers=2)    
    
    return trainloader,testloader
#%%
def eval_net_new(net,loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return (correct / total)
#%%    
#best choice from part2: e=16,b=8,l=0.001
def train_net(e,b,l):
    trainloader,testloader = mod_batch(b)    # set batch size here
    tr_loss = []
    test_acc = []
    net = torchvision.models.resnet101(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=l, momentum=0.9)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(e):  # loop over the dataset multiple times
        tloss = 0
        running_loss = 0.0
        #scheduler.step(epoch)
        start = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            # print statistics
            running_loss += loss.item()
        end = time.time()  
           
#        for data in valloader:
#            images, labels = data
#            outputs = net(images)
#            loss = criterion(outputs, labels)
#            vloss += loss.item()
    
        tr_loss.append(tloss/len(trainloader))
#        val_loss.append(vloss/len(valloader))
        
#        val_acc.append(eval_net_new(net,valloader))
        test_acc.append(eval_net_new(net,testloader))
 
        print ('epoch %d finished, train loss: %.3f' 
               % (epoch + 1, tloss/len(trainloader)))
        print ('time period: %.3f' % (end - start))
    print('Finished Training')
    
    torch.save(net.state_dict(), 'resnet101.pth')
    return tr_loss,test_acc

#%%
def plot(tr_loss,test_acc):
    plt.plot(tr_loss,'r')     # the color is set by default, you can also set your own color
#    plt.plot(val_loss,'g') # lw controls the line width
    plt.legend(['train', 'val'])
    plt.title('loss curve')
    plt.xlabel('epoch')
    plt.ylabel('loss rate')
    plt.show()

#    plt.plot(val_acc,'b')
    plt.plot(test_acc,'k')
    plt.legend(['val','test'])
    plt.title('accuracy curve')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
#%%
if __name__ == '__main__':

    tr_loss,test_acc = train_net(8,8,0.001)   #de tr
    plot(tr_loss,test_acc)
    print (test_acc)
#    print (val_acc)
