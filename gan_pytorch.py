#!/usr/bin/env python

# Generative Adversarial Networks (GAN) example in PyTorch. Tested with PyTorch 0.4.1, Python 3.6.7 (Nov 2018)
# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
DEBUG=True
matplotlib_is_available = True
try:
  from matplotlib import pyplot as plt
except ImportError:
  print("Will skip plotting; matplotlib is not available.")
  matplotlib_is_available = False
# Data params
data_mean = 4 #真實資料的平均值 用來透過np.random.normal生成高斯分布
data_stddev = 1.25 #真實資料的標準差 用來透過np.random.normal生成高斯分布

# ### Uncomment only one of these to define what data is actually sent to the Discriminator
#(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
#(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)
#(name, preprocess, d_input_func) = ("Data and diffs", lambda data: decorate_with_diffs(data, 1.0), lambda x: x * 2)
(name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)

print("Using data [%s]" % (name))

# ##### DATA: Target data and generator input data

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian
    #高斯分布
def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian
    #生成size=(m,n)的uniform分布tensor

# ##### MODELS: Generator model and discriminator model

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.first_froward=True
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f
    def forward(self, x):
        if self.first_froward and DEBUG:
            self.first_froward=False
            return self.firstForward(x)
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x
    #To Check Network Size When Debug Mode
    def firstForward(self,x):
        #測試用
        print('---Generator---')
        print('input size : ',x.size())
        x = self.map1(x)
        x = self.f(x)
        print('hidden1 size : ',x.size())
        x = self.map2(x)
        x = self.f(x)
        print('hidden2 size : ',x.size())
        x = self.map3(x)
        print('output size : ',x.size())
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.first_froward=True
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f
    def firstForward(self,x):
        #測試用
        print("---Discriminator---")
        print('input size:',x.size())
        x = self.f(self.map1(x))
        print('hidden 1 size:',x.size())
        x = self.f(self.map2(x))
        print('hidden 2 size:',x.size())
        x=self.f(self.map3(x))
        print('output size',x.size())
        return x
    def forward(self, x):
        if self.first_froward and DEBUG:
            self.first_froward=False
            return self.firstForward(x)
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))
        

def extract(v):
    """
    將v tensor中的東西解開成list回傳
    v: Tensor([float])
    Return : List[]
    """ 
    #print("v : ",v,"\tv storage:",v.data.storage().tolist())   
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def get_moments(d):
    # Return the first 4 moments of the data provided
    mean = torch.mean(d)
    #平均 Ex
    diffs = d - mean
    #所有項目跟平均的差 (輸入Tensor(1,500))
    
    #假設
    """
    d=Tensor(
    [
        [1],
        [2],
        [3],
        ...,
        [500]
    ])
    mean=10
    """
    #則
    """
    d-mean=Tensor(
        [
            [-9],
            [-8],
            [-7],
            ...,
            [490]
        ]      
    )
    """
    #相當於 [[xi] - mean for xi in p] 也就是數學上 所有數字跟平均的差
 
    
    var = torch.mean(torch.pow(diffs, 2.0))
    #把所有差平方再取平均，就變成變異數Variance
    std = torch.pow(var, 0.5)
    #標準差 standard deviation，變異數開根號
    #不過其實torch有提供torch.std 只是他要用到diffs之類的所以才會自己算
    zscores = diffs / std
    #標準分數 a.k.a. 標準化值 z-score a.k.a. standard score
    skews = torch.mean(torch.pow(zscores, 3.0))
    #偏度 Skewness 偏度衡量實數隨機變量機率分布的不對稱性
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
    #kurtosis的複數 表示PDF在平均附近增減的趨勢 可以翻譯為陡峭的程度
    final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))
    return final

def decorate_with_diffs(data, exponent, remove_raw_data=False):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    if remove_raw_data:
        return torch.cat([diffs], 1)
    else:
        return torch.cat([data, diffs], 1)

def train():
    # Model parameters
    g_input_size = 1      # Random noise dimension coming into generator, per output vector
    g_hidden_size = 5     # Generator complexity
    g_output_size = 1     # Size of generated output vector
    d_input_size = 500    # Minibatch size - cardinality of distributions 
    #注意到，這個是實際資料的長度，因為傳遞給D的前置處理可以切換，所以實際網路的輸入size不是這個
    #需在前置處理裡面定義處理d_input_size的d_input_func(d_input_size)
    d_hidden_size = 10    # Discriminator complexity
    d_output_size = 1     # Single dimension for 'real' vs. 'fake' classification
    minibatch_size = d_input_size

    d_learning_rate = 1e-3 # 10^-3 = 0.001
    g_learning_rate = 1e-3
    sgd_momentum = 0.9

    num_epochs = 5000
    print_interval = 100
    d_steps = 20
    g_steps = 20

    dfe, dre, ge = 0, 0, 0 #d real error, d fack error, g error
    d_real_data, d_fake_data, g_fake_data = None, None, None

    discriminator_activation_function = torch.sigmoid
    generator_activation_function = torch.tanh

    d_sampler = get_distribution_sampler(data_mean, data_stddev)
    gi_sampler = get_generator_input_sampler()
    G = Generator(input_size=g_input_size,
                  hidden_size=g_hidden_size,
                  output_size=g_output_size,
                  f=generator_activation_function)
    D = Discriminator(input_size=d_input_func(d_input_size),
                      hidden_size=d_hidden_size,
                      output_size=d_output_size,
                      f=discriminator_activation_function)
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
    g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)

    for epoch in range(num_epochs):#就訓練那麼多次 沒有特殊終止條件
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()
            #清空梯度累積值

            #  1A: Train D on real
            d_real_data = Variable(d_sampler(d_input_size))# N=500的高斯分布(長度為500) size(1,500)
            
            d_real_decision = D(preprocess(d_real_data))#先前置處理(預設使用 高斯分布的幾個數值 mean std zscore等 來訓練D)
            d_real_error = criterion(d_real_decision, Variable(torch.ones([1,1])))  # ones = true 應該是將決策的跟Tensor([[1]])目標來計算loss (透過BCE) y , y'計算
            d_real_error.backward() # compute/store gradients, but don't change params
            #透過真實資料訓練D，累積梯度
            #  1B: Train D on fake
            d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            
            d_fake_decision = D(preprocess(d_fake_data.t()))
            #torch.t or tensor.t -> tensor  即matrix transpose運算 ->交換row、col
            #size(500,1)->(1,500)
            """
            d_fack_data.size() # -> ()
            tensor(
            [
                #    V Col1
                [-0.3821], # <-Row1
                [-0.3846], # <-Row2
                [-0.4510], # ...
                ...,
                [-0.1234] #<- RowN
            ])
            ----via method t()--->
            tensor(
            [      #Col1V      Col2V ...      ColN V
                [-0.3821, -0.3846, ... , -0.1234] <- Row1
            ])
            """
            #這裡是透過G來生假資料訓練D得到假的Loss，理論上y=0(因為是假的資料)即 從真實資料來的機率判斷為0
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1,1])))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

            dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]#因為回傳的List都只有一個項目 所以取[0]得到element

        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = Variable(gi_sampler(minibatch_size, g_input_size))#產生(minib,g_size)的uniform雜訊
            g_fake_data = G(gen_input)#透過雜訊生成假樣本
            dg_fake_decision = D(preprocess(g_fake_data.t()))#transpose並前置處理提取四項機率數值到D給D判斷
            #只透過D來訓練G
            g_error = criterion(dg_fake_decision, Variable(torch.ones([1,1])))  # Train G to pretend it's genuine 算Loss

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters
            ge = extract(g_error)[0]

        if epoch % print_interval == 0:
            print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
                  (epoch, dre, dfe, ge, stats(extract(d_real_data)), stats(extract(d_fake_data))))

    if matplotlib_is_available:
        print("Plotting the generated distribution...")
        values = extract(g_fake_data)
        print(" Values: %s" % (str(values)))
        plt.hist(values, bins=50)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogram of Generated Distribution')
        plt.grid(True)
        plt.show()


train()
