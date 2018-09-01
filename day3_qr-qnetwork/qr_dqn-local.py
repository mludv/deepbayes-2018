
# coding: utf-8

# # Distributional Reinforcement Learning with Quantile Regression

# ![QR-Net1](https://ars-ashuha.github.io/images/QR-Net1.png)
# ![QR-Net2](https://ars-ashuha.github.io/images/QR-Net2.png)
# ![Vis](https://ars-ashuha.github.io/images/vis.png)
# 
# - Distributional Reinforcement Learning with Quantile Regression, https://arxiv.org/pdf/1710.10044.pdf 
# - The solution is you got stuck you could cheat a little bit https://github.com/ars-ashuha/quantile-regression-dqn-pytorch 

# # Implementation

# In[1]:


import gym
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from logger import Logger
from rl_utils import ReplayMemory, huber


# In[2]:


class Network(nn.Module):
    def __init__(self, len_state, num_quant, num_actions):
        nn.Module.__init__(self)
        
        self.num_quant = num_quant
        self.num_actions = num_actions
        
        self.layer1 = nn.Linear(len_state, 255)
        self.layer2 = nn.Linear(255, num_quant*num_actions)
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        return x.view(-1, self.num_actions, self.num_quant)
    
    def select_action(self, state, eps):
        if not isinstance(state, torch.Tensor): 
            state = torch.Tensor([state])    
            
        action = torch.randint(0, self.num_actions, (1,))
        if random.random() > eps:
            action = self.forward(state).mean(2).max(1)[1]
        return int(action)


# In[3]:


a = [[[0, 0],[1,1],[2,2]], [[3,3],[4,4],[5,5]], [[6,6],[7, 7], [8,8]]]
a = torch.tensor(a)
a[[0,1, 2], [1,2,0]]


# In[4]:


# Here we've defined a schedule for exploration i.e. random action with prob eps
eps_start, eps_end, eps_dec = 0.9, 0.1, 500 
eps = lambda steps: eps_end + (eps_start - eps_end) * np.exp(-1. * steps / eps_dec)


# In[5]:


# We start from CartPole-v0         
# and then will solve MountainCar-v0
env_name = 'CartPole-v0' 
env = gym.make(env_name)

memory = ReplayMemory(10000)
logger = Logger('q-net', fmt={'loss': '.5f'})


# In[6]:


Z = Network(len_state=len(env.reset()), num_quant=2, num_actions=env.action_space.n)
Ztgt = Network(len_state=len(env.reset()), num_quant=2, num_actions=env.action_space.n)
optimizer = optim.Adam(Z.parameters(), 1e-3)


# In[7]:


tau = torch.Tensor((2 * np.arange(Z.num_quant) + 1) / (2.0 * Z.num_quant)).view(1, -1)
tau


# ## Training cicle

# In[8]:


gamma, batch_size = 0.99, 32 
steps_done, running_reward = 0, None

for episode in range(501): 
    sum_reward = 0
    state = env.reset()
    while True:
        steps_done += 1
        
        action = Z.select_action(torch.Tensor([state]), eps(steps_done))
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, next_state, reward, float(done))
        sum_reward += reward
        
        if len(memory) < batch_size: break    
            
        states, actions, rewards, next_states, dones = memory.sample(batch_size)
        
        theta = Z(states)[torch.arange(batch_size), actions, :]
        assert theta.shape == (batch_size, Z.num_quant), f"{theta.shape} != {(batch_size, Z.num_quant)}"
        
        
        # Calculate quantiles for the next stage with target network 
        # and then take value for a max action
        Znext = Ztgt(next_states).detach()
        Znext_max = Znext[torch.arange(batch_size), Znext.mean(2).max(1)[1]]
        assert Znext_max.shape == (batch_size, Z.num_quant), f"{Znext_max.shape} != {(batch_size, Z.num_quant)}"
                                       
        Ttheta = rewards + gamma * (1 - dones) * Znext_max
        # Calculate loss, use this trick to compute pairwise differences
        # Trick Tensor of shape (3,2,1) minus Tensor of shape (1,2,3) is Tensor of shape (3, 2, 3)
        # With all pairwise differences :)
        # Use Huber elementwise function to compute Huber loss
        diff = Ttheta.t().unsqueeze(-1) - theta 
        loss = torch.mean(huber(diff) * (tau - (diff.detach() < 0).float()).abs())
        ###########################################################

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = next_state
        
        if steps_done % 100 == 0:
            Ztgt.load_state_dict(Z.state_dict())
            
        if done and episode % 50 == 0:
            logger.add(episode, steps=steps_done, running_reward=running_reward, loss=loss.data.numpy())
            logger.iter_info()
            
        if done: 
            running_reward = sum_reward if not running_reward else 0.2 * sum_reward + running_reward*0.8
            break


# # Vizualization

# ## Train the model for  MountainCar-v0 env here!

# In[10]:


import time
import matplotlib.pyplot as plt
from IPython import display

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')

from matplotlib import rcParams
rcParams['figure.figsize'] = 7, 2
rcParams['figure.dpi'] = 150


# In[11]:


actions={
    'CartPole-v0': ['Left', 'Right'],
    'MountainCar-v0': ['Left', 'Non', 'Right'],
}


# In[12]:


def get_plot(q):
    eps, p = 1e-8, 0
    x, y = [q[0]-np.abs(q[0]*0.2)], [0]
    for i in range(0, len(q)):
        x += [q[i]-eps, q[i]]
        y += [p, p+1/len(q)]
        p += 1/len(q)
    x+=[q[i]+eps, q[i]+np.abs(q[i]*0.2)]
    y+=[1.0, 1.0]
    return x, y


# In[13]:


state, done, steps = env.reset(), False, 0
while True:
    plt.clf()
    steps += 1
    action = Z.select_action(torch.Tensor([state]), eps(steps_done))
    state, reward, done, _ = env.step(action)
    
    if steps % 3 == 0:  
        plt.subplot(1, 2, 1)
        plt.title('step = %s' % steps)
        plt.imshow(env.render(mode='rgb_array'))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        Zval = Z(torch.Tensor([state])).detach().numpy()
        for i in range(env.action_space.n):
            x, y = get_plot(Zval[0][i])
            plt.plot(x, y, label='%s Q=%.1f' % (actions[env_name][i], Zval[0][i].mean()))
            plt.legend(bbox_to_anchor=(1.1, 1.1), ncol=3, prop={'size': 6})

        if done: break
        display.clear_output(wait=True)
        display.display(plt.gcf())
plt.clf()

