#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pymoo.interface import crossover
from pymoo.factory import get_crossover
from pymoo.interface import mutation
from pymoo.factory import get_mutation
import random
import deap
from deap import base
from deap import creator
from deap import tools


# In[2]:


def sbx_real(parents,prob,eta_c):
    k=0
    child=np.full((parents.shape),np.nan)
    parents_=parents.copy()
    parents_.tolist()
    for i in range(int(parents.shape[0]/2)):
        if random.random()<prob:
            child[k]=(np.array(deap.tools.cxSimulatedBinary(parents_[k], parents_[k+1], eta_c)))[0]
            child[k+1]=(np.array(deap.tools.cxSimulatedBinary(parents_[k], parents_[k+1], eta_c)))[1]
            k=k+2
        else:
            child[k]=parents_[k]
            child[k+1]=parents_[k+1]
            k=k+2
    return child                    


# In[3]:


def sbx_int(parents,prob,eta_c,xl,xu):
    k=0

    if parents.shape[0]==parents.size:
        parents=parents.reshape(-1,1)
        child=np.full((parents.shape),np.nan)
        for i in range(int(parents.shape[0]/2)):
            child[k]  =(crossover(get_crossover("int_sbx", prob=prob, eta=eta_c, prob_per_variable=1.0),parents[k].reshape(1,-1),parents[k+1].reshape(1,-1) , xl=xl, xu=xu))[0]
            child[k+1]=(crossover(get_crossover("int_sbx", prob=prob, eta=eta_c, prob_per_variable=1.0),parents[k].reshape(1,-1),parents[k+1].reshape(1,-1) , xl=xl, xu=xu))[1]
            k=k+2
    else:
        child=np.full((parents.shape),np.nan)
        for i in range(int(parents.shape[0]/2)):
            child[k]  =(crossover(get_crossover("int_sbx", prob=prob, eta=eta_c, prob_per_variable=1.0),parents[k].reshape(1,-1),parents[k+1].reshape(1,-1) , xl=xl, xu=xu))[0]
            child[k+1]=(crossover(get_crossover("int_sbx", prob=prob, eta=eta_c, prob_per_variable=1.0),parents[k].reshape(1,-1),parents[k+1].reshape(1,-1) , xl=xl, xu=xu))[1]
            k=k+2
    return child


# In[4]:


def mut_poly(n,eta,low,up,prob):
    
    if n.shape[0]==n.size:
        n=n.tolist()
        mut=np.array(deap.tools.mutPolynomialBounded([m[0] for m in n], eta, low, up, prob))
    else: 
        p=n.shape[1]
        mut=np.full((n.shape),np.full)
        n=n.tolist()
        for i in range(p):
            mut[:,i]=np.array(deap.tools.mutPolynomialBounded([m[i] for m in n], eta, low, up, prob))
    return mut


# In[5]:


def mut_int(x,eta,low,up,prob):
    
    if x.shape[0]==x.size:
        x=x.reshape(-1,1)
        mut=np.full((x.shape),np.nan)
        mut=mutation(get_mutation("int_pm", eta=eta, prob=prob), x , xl=low, xu=up)
    else:
        mut=np.full((x.shape),np.nan)
        for i in range((x.shape[1])):
            mut[:,i]=mutation(get_mutation("int_pm", eta=eta, prob=prob), x[:,i].reshape(1,-1) , xl=low, xu=up)
    return mut


# In[6]:


def mut_flip(n,prob):
    
    if n.shape[0]==n.size:
        n=n.tolist()
        mut=np.array(deap.tools.mutFlipBit([m[0] for m in n], eta, low, up, prob))
    else:
        p=n.shape[1]
        mut=np.full((n.shape),np.nan)
        n=n.tolist()
        for i in range(p):
            mut[:,i]=np.array((deap.tools.mutFlipBit([m[i] for m in n], prob)))
    return mut


# In[7]:


def tournament_selection(pop, fitness):
    
    fitness=fitness.reshape((fitness.shape[0],1))
    pop_=np.array(np.concatenate((pop,fitness),axis=1))
    pop1_=np.array(pop_.copy())
    np.random.shuffle(pop1_)
    a=np.array(pop1_[pop1_[:,-1]>=pop_[:,-1]])
    b=np.array(pop_[pop_[:,-1]>pop1_[:,-1]])
    par=np.array(np.concatenate((a,b),axis=0))
    par=par[par[:,-1].argsort()][::-1]
    return (par[:,:-1])


# In[ ]:




