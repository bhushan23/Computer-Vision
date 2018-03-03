
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as tutils


# In[2]:


batch_size = 100
path = './data/genImg/'


# In[3]:


# MNIST dataset
dataset = datasets.MNIST(root='./data',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)


# In[4]:


IS_CUDA = False
if torch.cuda.is_available():
    IS_CUDA = True
    
def var(x):
    if IS_CUDA:
        x = x.cuda()
    return Variable(x)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# In[5]:


Generator_input = 64
Generator = nn.Sequential(
        nn.Linear(Generator_input, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 784),
        nn.Tanh())

Discriminator = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256,1),
        nn.Sigmoid())

if IS_CUDA:
    Discriminator.cuda()
    Generator.cuda()


# In[6]:


lossCriterion = nn.BCELoss()
G_opt = torch.optim.Adam(Generator.parameters(), lr = 0.0001)
D_opt = torch.optim.Adam(Discriminator.parameters(), lr = 0.0001)
fixed_x = var(torch.randn(batch_size, Generator_input))


# In[7]:


outputImages = []
def train(num_epochs = 10, d_iter = 1):
    for epoch in range(num_epochs):
        for data in data_loader:
            image, _  = data
            image = var(image.view(image.size(0),  -1))
            # Train Discriminator
            #for k in range(0, d_iter):
            # For Log D(x)
            D_real = Discriminator(image)
            # For Log(1 - D(G(Z)))
            Z_noise = var(torch.randn(batch_size, Generator_input))
            G_fake = Generator(Z_noise)
            D_fake = Discriminator(G_fake)

            # Calculate Discriminator Loss
            D_real_loss = lossCriterion(D_real, Variable(torch.ones(batch_size, 1)))
            D_fake_loss = lossCriterion(D_fake, Variable(torch.zeros(batch_size, 1)))
            D_loss = D_real_loss + D_fake_loss

            # Backprop Discriminator
            Discriminator.zero_grad()
            D_loss.backward()
            D_opt.step()

                
            # Train Generator
            Z_noise = var(torch.randn(batch_size, Generator_input))
            G_fake = Generator(Z_noise)
            D_fake = Discriminator(G_fake)
            # Compute Generator Loss
            G_loss = lossCriterion(D_fake, Variable(torch.ones(batch_size, 1)))
            
            # Backprop Genearator
            Discriminator.zero_grad()
            Generator.zero_grad()
            G_loss.backward()
            G_opt.step()
            
        #print epoch
        print 'Epoch [{}/{}], Discriminator {:.4f}, Generator {:.4f}'.format(epoch+1, num_epochs, D_loss.data[0], G_loss.data[0])
        pic = Generator(fixed_x)
        pic = pic.view(pic.size(0), 1, 28, 28) 
        outputImages.append(pic)
        torchvision.utils.save_image(pic.data.cpu(), path+'/image_{}.png'.format(epoch))             


# In[8]:


train(200)


# In[13]:


torch.save(Generator.state_dict(), './Generator.pkl')
torch.save(Discriminator.state_dict(), './Discriminator.pkl')

