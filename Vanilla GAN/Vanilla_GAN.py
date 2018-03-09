
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
import imageio


# In[2]:


# Options being used 
batch_size = 100
imgDim = 28
path = './genImg/'
showPlot = False
savePlot = True
num_epochs = 200
IS_CUDA = False

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5))])


# In[3]:


# MNIST dataset
dataset = datasets.MNIST(root='./data',
                         train=True,
                         transform=transform,
                         download=True)
# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)


# In[4]:


# Helper routines
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

def generate_animation(root, epoch, name):
    images = []
    for e in range(epoch):
        img_name = root+'/image_'+str(e+1)+'.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(root+ '/' + name +'.gif', images, fps=5)


# In[5]:


# Network
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
G_opt = torch.optim.Adam(Generator.parameters(), lr = 0.0002)
D_opt = torch.optim.Adam(Discriminator.parameters(), lr = 0.0002)
fixed_x = var(torch.randn(batch_size, Generator_input))


# In[9]:


outputImages = []
D_loss_plot = []
G_loss_plot = []
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
            D_real_loss = lossCriterion(D_real, var(torch.ones(batch_size, 1)))
            D_fake_loss = lossCriterion(D_fake, var(torch.zeros(batch_size, 1)))
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
            G_loss = lossCriterion(D_fake, var(torch.ones(batch_size, 1)))
            
            # Backprop Genearator
            Discriminator.zero_grad()
            Generator.zero_grad()
            G_loss.backward()
            G_opt.step()
            
        #print epoch
        print 'Epoch [{}/{}], Discriminator {:.4f}, Generator {:.4f}'.format(epoch+1, num_epochs, D_loss.data[0], G_loss.data[0])
        D_loss_plot.append(D_loss.data[0])
        G_loss_plot.append(G_loss.data[0])
        pic = Generator(fixed_x)
        pic = pic.view(pic.size(0), 1, 28, 28) 
        pic = denorm(pic.data)
        outputImages.append(pic)
        torchvision.utils.save_image(pic, path+'/image_{}.png'.format(epoch))             


# In[10]:


# num_epochs = 200
train(num_epochs)


# In[22]:


# Plot the Loss for Generator and Discriminator
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Vanilla GAN Loss')
plt.plot(D_loss_plot)
plt.plot(G_loss_plot)

if showPlot:
    plt.show()
if savePlot:
    plt.savefig('Loss_Plot_Vanilla_GAN_'+str(num_epochs)+'.png')


# In[28]:


# Generate GIF
generate_animation(path, 5, 'Vanilla_Gan')


# In[29]:


# Save the model
torch.save(Generator.state_dict(), './Generator.pkl')
torch.save(Discriminator.state_dict(), './Discriminator.pkl')


# In[2]:


# Load model for debugging and testing
#Generator.load_state_dict(torch.load('Generator200.pkl'))
#Discriminator.load_state_dict(torch.load('Discriminator200.pkl'))

