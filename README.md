Generative Adversarial Networks (GANs) in PyTorch
===============


### Introduction


See https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9 for the relevant blog post.


### Running
Run the sample code by typing:


```
./gan_pytorch.py
```

...and you'll train two nets to battle it out on a shifted/scaled Gaussian distribution. The 'fake' distribution should match the 'real' one within a reasonable time.

GAN in 50 lines analyze
===
###### tags: `專題` `Neural Network` `GAN`
### [原始文章中文翻譯](https://www.pytorchtutorial.com/50-lines-of-codes-for-gan/)
### [原始GitHub](https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py)
### [我打過註解的Github](https://github.com/NawaNae/pytorch-generative-adversarial-networks)
## 目標
生成一個高斯分布陣列(1維)作為真實資料，讓Generator在合理的時間內生出假的且合理的高斯分布。
可以說他在用G來耦合np.random.normal(mu,sigma)
## 文章符號
R：原始、真實數據集 
I：作為熵的一項來源，進入生成器的隨機噪音
G：生成器，試圖模仿原始數據
D：判別器，試圖區別 G 的生成數據和 R

## Code
### 參數
#### data_mean = 4
平均值(用在高斯函數生成取樣器資料分布)
#### data_stddev = 1.25
標準差(用在高斯函數生成取樣器資料分布)
#### g_input_size = 1
Generator的輸入大小
#### g_hidden_size = 5
Generator的隱藏層大小(複雜度)
#### g_output_size = 1
Generator輸出大小
#### d_input_size = 500  
資料的大小
實際輸入Discriminator的大小(輸入層大小)由前置處理決定，透過d_input_func(d_input_size)計算得到
#### d_hidden_size = 10
Discriminator的隱藏層大小
#### d_output_size = 1
Discriminator輸出層大小，為1的純量，因為輸出為機率
#### d_learning_rate = 1e-3 # 10^-3 = 0.001
Discriminator學習率
#### g_learning_rate = 1e-3
Generator學習率
#### num_epochs = 5000
epochs次數
#### print_interval = 100
印出分布圖的頻率(每X次)
#### d_steps = 20
一個epoch要做幾次訓練discriminator
#### g_steps = 20
一個epoch要做幾次訓練generator
#### dfe
discriminator fake error
discriminator真實資料訓練時的error
#### dre
discriminator real error
discriminator假資料訓練時的error
#### ge
generator error
#### criterion = nn.BCELoss()
BCELoss的實體，用來計算Grandient
#### discriminator_activation_function = torch.sigmoid
discriminator用的激勵函數(sigmoid)
猜測是因為其輸出是機率，所以範圍要控制在0~1之間，且極端值控制在趨近於1與0
![](https://i.imgur.com/GVbSVj7.png)
#### generator_activation_function = torch.tanh
Generator用的激勵函數(tanh)
跟D同類型，猜測沒強迫輸出為機率(0~1)所以採用正負平衡的tanh，壓縮也比較少
![](https://i.imgur.com/fkJzu3k.png)

### 解析
#### R
真實資料，因為目標要做高斯分布，這裡直接用np.random.normal來產生高斯分布，程式碼則透過取樣器即其生成函數來實現
取樣器生成 透過下列函數輸入 平均、標準差，可以生成高斯分布的取樣器函數，生成(1,n)維高斯取樣器
```Python
def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian
```
#### I
Generator的雜訊，透過rand(uniform分布)取樣，這樣才不能透過縮放平移來達到高斯分布(增加難度)，需透過非線性方式擬合
程式碼中生成取樣器函數決定生成的Size，並回傳random Uniform取樣器
```Python
def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian
    #生成size=(m,n)的uniform分布tensor
```
#### G
Generator透過輸入雜訊，扭曲分布空間，來達到模擬分布
程式碼中採用Linear，兩個隱藏層的做法(MLP)
1. 這樣才可以擬合非線性可分資料(MLP)
2. 預設採用tanh作為激勵函數
3. 採用BCELoss計算Gradient
4. 中間隱藏層大小預設較D還要低(5) (複雜度低)
##### 網路圖
![](https://nawanae.github.io/pytorch-generative-adversarial-networks/images/generator.png)
###### 過程大小變化
![](https://i.imgur.com/kBDyGya.png)
##### code
```Python
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.first_froward=True
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f
    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x
```
#### D
Discriminator 將被前置處理過的資料作為輸入，通過2層隱藏層的MLP輸出一個機率(輸出Size(1))
1. 這樣才可以擬合非線性可分資料(MLP)
2. 預設採用sigmoid作為激勵函數(大概是因為機率為0~1之間的數字)
3. 採用BCELoss計算Gradient
4. 中間隱藏層大小預設較G還要高(10) (複雜度高)
##### 網路圖
![](https://nawanae.github.io/pytorch-generative-adversarial-networks/images/discriminator.png)
###### 過程大小變化
![](https://i.imgur.com/IkAzmYh.png)
##### code
```Python
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.first_froward=True
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f
    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))
```
### function解析
### train() 
##### 步驟
1. 初始化參數、BCELoss、Generator、Discriminator、optimizer等
2. epoch迴圈(num_epoch次)
    1. d step迴圈(執行d的step次)
        ##### 訓練Discriminator
        ###### 透過真實樣本
        1. 清空D的梯度
        2. 取得真實資料並前置處理
        3. 將處理後的資料帶入D取得y(是真實資料的機率)
        4. 利用BCE透過剛剛的y與y'=1(真實資料來的機率)計算取得loss
        5. 計算gradient(backward)並累計
        ###### 透過Generator生成假樣本
        6. 取得G隨機生成樣本並前置處理
        7. 將處理後的資料帶入D取得y(是真實資料的機率)
        8. 利用BCE透過剛剛的y與y'=0(從G來的機率)計算取得loss
        9. 計算gradient(backward)並累計
        ###### 其他
        10. 更新參數
    2. g step迴圈(執行g的step次)
        ##### 訓練Generator
        ###### 僅透過Discriminator
        1. 清空G的Gradient
        2. 從Uniform雜訊取樣器取得雜訊
        3. 將取得雜訊帶入G做成假樣本y
        4. 將假樣本前置處理後代入D取得機率(G的目標是讓D的機率為1(都從真實資料來))
            ==這裡透過z=D(y) 映射y到D的輸出上==
        5. 利用BCE透過剛剛的d算出來的機率z跟z'=1(G預設要讓D以為都是真的，所以為1)
        6. 計算gradient(backward)並累計
        ###### 其他
        7. 更新參數
### preprocess(data:Tensor)->data:Tensor
透過自訂的function在進行D判斷前先做處理，可以指到實際上處理的function，預設為get_monents
### get_monents(data:Tensor)->data:Tensor
回傳data的Tensor([mean, standard deviation, standard score, kurtoses])作為D的輸入
### stats(data:Tensor)->list
回傳data的[mean, standard deviation]
### extract(vector:Tensor)->list
回傳vector(原本是Tensor)的內部list
就是解開Tensor的包裝

# 筆記
* ![](https://i.imgur.com/uxQjOV6.png)


# Ref
[torch新手村](https://medium.com/pyladies-taiwan/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E6%96%B0%E6%89%8B%E6%9D%91-pytorch%E5%85%A5%E9%96%80-511df3c1c025)
