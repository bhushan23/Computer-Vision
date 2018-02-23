
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision


# In[2]:


# MNIST dataset
dataset = datasets.MNIST(root='./data',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=100, 
                                          shuffle=True)


# In[3]:


IS_CUDA = False
if torch.cuda.is_available():
    IS_CUDA = True


# In[4]:


def var(x):
    if IS_CUDA:
        x = x.cuda()
    return Variable(x)


# In[5]:


# Model 1
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim = 400, z_dim = 20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2))
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid())
    
    def reparameterize(self, mu, log_var):
        epsilon = var(torch.randn(mu.size(0), mu.size(1)))
        z = mu + epsilon * torch.exp(log_var/2)
        return z
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim = 1)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var
    
    def sample(self, z):
        return self.decoder(z)
    
vae = VAE()
if IS_CUDA:
    vae.cuda()


# In[6]:


optimizer = torch.optim.Adam(vae.parameters(), lr = 0.001)


# In[9]:


data_iter = iter(data_loader)
fixed_x,_ = next(data_iter)
torchvision.utils.save_image(fixed_x.cpu(), './data/real_images.png')
fixed_x = var(fixed_x.view(fixed_x.size(0), -1))


# In[15]:


num_epochs = 100


# In[20]:


outputImages = []
for epoch in range(num_epochs):
    for data in data_loader:
        img, _ = data
        img = var(img.view(img.size(0), -1))
        out, mu, log_var = vae(img)
        rc_loss = F.binary_cross_entropy(out, img, size_average=False)
        KL_div = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))
        
        total_loss = rc_loss + KL_div
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    print 'Epoch [{}/{}], Loss {:.4f}, Entropy: {:.4f}, KL: {: .4f} '.format(epoch+1, num_epochs, total_loss.data[0], rc_loss.data[0], KL_div.data[0]                                                                 )
    pic, _, _ = vae(fixed_x)
    pic = pic.view(pic.size(0), 1, 28, 28) 
    outputImages.append(pic)
    torchvision.utils.save_image(pic.data.cpu(), './data/image_{}.png'.format(epoch))

