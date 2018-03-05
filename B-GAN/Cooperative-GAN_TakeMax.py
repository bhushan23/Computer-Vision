
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
import copy
from PIL import Image


# In[2]:


batch_size = 100
imgDim = 28
path = './genImg/'


# In[3]:


# MNIST dataset
dataset = datasets.MNIST(root='../Data/MNIST',
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
    npimg = img.data.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def save_image(pic, path):
    grid = torchvision.utils.make_grid(pic.data, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


# In[5]:


Generator_input = 64
'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = x.resize(batch_size, 1, imgDim, imgDim)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(x),2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.sigmoid(x)
'''
    
    
D = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256,1),
        nn.Sigmoid())

Generator_input = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(Generator_input, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Sequential(nn.Linear(256, 784), nn.Tanh())
        self.lR = nn.LeakyReLU(0.2)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.lR(x)
        x = self.fc2(x)
        x = self.lR(x)
        return self.fc3(x)
'''

Generator = nn.Sequential(
        nn.Linear(Generator_input, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 784),
        nn.Tanh())

'''
#D = Discriminator()

# Create n Generators
#Gen = copy.deepcopy(Generator)
Gen = Generator()
Generators = []
GOptimizers = []
NumberOfGenerators = 3
for i in range(NumberOfGenerators):
    #Generators.append(copy.deepcopy(Generator))
    Generators.append(Generator())

if IS_CUDA:
    D.cuda()
    G.cuda()
    for each in Generators:
        each.cuda()


# In[6]:


lossCriterion = nn.BCELoss()
D_opt = torch.optim.Adam(D.parameters(), lr = 0.0001)
G_opt = torch.optim.Adam(Gen.parameters(), lr = 0.0001)
GOptimizers = []
for each in Generators:
    GOptimizers.append(torch.optim.Adam(each.parameters(), lr = 0.0001))

fixed_x = var(torch.randn(batch_size, Generator_input))

GeneratorLoss = []
def backPropGenerator(index, GeneratorLoss):
    Generators[index].zero_grad()
    GeneratorLoss[index].backward()
    GOptimizers[index].step()



# In[ ]:


'''
outputImages = []
BestPerformingGenerator = 0
def train(Gen, BestPerformingGenerator, num_epochs = 10, d_iter = 1):
    for epoch in range(num_epochs):
        for data in data_loader:
            image, _  = data
            image = var(image.view(image.size(0),  -1))
            
            #Gen = copy.deepcopy(Generators[BestPerformingGenerator])
            
            # Train Discriminator
            #for k in range(0, d_iter):
            # For Log D(x)
            #print image.shape
            D_real = D(image)
            # For Log(1 - D(G(Z)))
            Z_noise = var(torch.randn(batch_size, Generator_input))
            #print Z_noise.shape
            #print type(Gen)
            G_fake = Gen(Z_noise)
            #print G_fake.shape
            D_fake = D(G_fake)

            # Calculate Discriminator Loss
            D_real_loss = lossCriterion(D_real, var(torch.ones(batch_size, 1)))
            D_fake_loss = lossCriterion(D_fake, var(torch.zeros(batch_size, 1)))
            D_loss = D_real_loss + D_fake_loss

            # Backprop Discriminator
            D.zero_grad()
            D_loss.backward()
            D_opt.step()
   
            # Train Generators
            Z_noise = var(torch.randn(batch_size, Generator_input))
            G_fake = Gen(Z_noise)
            D_fake = D(G_fake)
            # Compute Generator Loss
            G_loss = lossCriterion(D_fake, var(torch.ones(batch_size, 1)))
            
            # Find best performing Generator
            GeneratorLoss = []
            lossList = []
            for each in Generators:
                Z_noise1 = var(torch.randn(batch_size, Generator_input))
                G_fake1 = each(Z_noise1)
                #print G_fake1.shape
                #print type(each)
                D_fake1 = D(G_fake1)
                # Compute Generator Loss
                G_loss1 = lossCriterion(D_fake1, var(torch.ones(batch_size, 1)))
                GeneratorLoss.append(G_loss1)
                lossList.append(float(G_loss1.data[0]))
            
            #print lossList
            #print type(lossList[0])
            BestPerformingGenerator = lossList.index(min(lossList))
            
            # Change other Generator with best performing ones config:
            Gen = copy.deepcopy(Generators[BestPerformingGenerator])
            G_opt = copy.deepcopy(GOptimizers[BestPerformingGenerator])

            # Backprop Genearator
            D.zero_grad()
            Gen.zero_grad()
            G_loss.backward()
            G_opt.step()
            
            backPropGenerator(BestPerformingGenerator, GeneratorLoss)
            
            for i in range(NumberOfGenerators):
                if i != BestPerformingGenerator:
                    Generators[i] = copy.deepcopy(Generators[BestPerformingGenerator])
                    GOptimizers[i] = copy.deepcopy(GOptimizers[BestPerformingGenerator])
                    backPropGenerator(i, GeneratorLoss)
                                
        #print epoch
        print 'Epoch [{}/{}], Discriminator {:.4f}, Generator {:.4f}'.format(epoch+1, num_epochs, D_loss.data[0], G_loss.data[0])
        pic = Gen(var(torch.randn(batch_size, Generator_input))) #(fixed_x)
        pic = pic.view(pic.size(0), 1, imgDim, imgDim) 
        outputImages.append(pic)
        torchvision.utils.save_image(pic.data, path+'image_{}.png'.format(epoch)) 
'''


# In[12]:


#train(Gen, 0, 10)


# In[7]:


outputImages = []
def train(BestPerformingGenerator, num_epochs = 10, d_iter = 1):
    for epoch in range(num_epochs):
        lossList = [0.0] * NumberOfGenerators
        for data in data_loader:
            image, _  = data
            image = var(image.view(image.size(0),  -1))
            
            #Gen = copy.deepcopy(Generators[BestPerformingGenerator])
            
            # Train Discriminator
            #for k in range(0, d_iter):
            # For Log D(x)
            #print image.shape
            #print 'Epoch [{}/{}].'.format(epoch+1, num_epochs)
            
            #for each in Generators:
            D_real = D(image)
            # For Log(1 - D(G(Z)))
            Z_noise = var(torch.randn(batch_size, Generator_input))
            #print Z_noise.shape
            #print type(Gen)
            G_fake = Generators[BestPerformingGenerator](Z_noise)
            #print G_fake.shape
            D_fake = D(G_fake)

            # Calculate Discriminator Loss
            D_real_loss = lossCriterion(D_real, var(torch.ones(batch_size, 1)))
            D_fake_loss = lossCriterion(D_fake, var(torch.zeros(batch_size, 1)))
            D_loss = D_real_loss + D_fake_loss

            # Backprop Discriminator
            D.zero_grad()
            D_loss.backward()
            D_opt.step()
                #print 'Discriminator Loop for: {}: {}'.format(i, D_loss.data[0])
   
            # Find best performing Generator
            i = 0
            GeneratorLoss = []
            for each, each_opt in zip(Generators, GOptimizers):
                Z_noise = var(torch.randn(batch_size, Generator_input))
                G_fake = each(Z_noise)
                #print G_fake1.shape
                #print type(each)
                D_fake = D(G_fake)
                # Compute Generator Loss
                G_loss = lossCriterion(D_fake, var(torch.ones(batch_size, 1)))
                GeneratorLoss.append(G_loss)
                lossList[i] += (float(G_loss.data[0]))
                i = i + 1
                D.zero_grad()
                each.zero_grad()
                G_loss.backward()
                each_opt.step()
                
                #backPropGenerator(i, GeneratorLoss)
                #print 'Generator Loop for: {}: {}'.format(i, G_loss.data[0])
            
            #print lossList
            #print type(lossList[0])
        BestPerformingGenerator = lossList.index(max(lossList)) # earlier was min
        print lossList
        for i in range(0, NumberOfGenerators):
            if i != BestPerformingGenerator:
                prev = Generators[i]
                Generators[i] = copy.deepcopy(Generators[BestPerformingGenerator])
                GOptimizers[i] = torch.optim.Adam(Generators[i].parameters(), lr = 0.0001)
                #copy.deepcopy(GOptimizers[BestPerformingGenerator])
                if Generators[i] == prev:
                    print 'SAME'

        #print epoch
        #print BestPerformingGenerator
        print 'Epoch [{}/{}], Discriminator {:.4f}, Best Generator[{}] {:.4f}'.format(epoch+1, num_epochs, D_loss.data[0], BestPerformingGenerator, GeneratorLoss[BestPerformingGenerator].data[0])
        pic = Generators[BestPerformingGenerator](var(torch.randn(batch_size, Generator_input))) #(fixed_x)
        pic = pic.view(pic.size(0), 1, imgDim, imgDim) 
        outputImages.append(pic)
        #torchvision.utils.save_image(pic.data, path+'image_{}.png'.format(epoch))   
        save_image(pic, path+'image_{}.png'.format(epoch))


# In[8]:


#train(0, 30)


# In[19]:


train(0, 40)


# In[9]:


# With optimizer on the fly
train(0, 40)


# In[8]:


# With optimizer on the fly and max loss
train(0, 30)


# In[9]:


#torch.save(G.state_dict(), './Generator.pkl')
#torch.save(D.state_dict(), './Discriminator.pkl')


# In[ ]:


#Generator.load_state_dict(torch.load('Generator200.pkl'))
#Discriminator.load_state_dict(torch.load('Discriminator200.pkl'))

