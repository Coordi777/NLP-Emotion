# NLP情感分类

## 环境

- Python环境下的Pytorch框架。
- 数据来源：Kaggle人工标注的数据集，分为训练集（共计5w条评论）与测试集（共计4w条评论）。
- 词向量数据集：glove.6B.100d.txt，来源于wiki百科和Gigaword数据集。

## 简介

​	循环神经网络（Recurrent Neural Network，RNN）与卷积神经网络不同，它是一种运算单元被循环使用的神经网络。对于一种语言来讲，其内容与上下文有关，一个词在不同的语境下会有不同的含义，而且输入的顺序也对结果有很大影响，即其结果不仅依赖于当前情况，而且与过去的情况有关。因此，对于NLP问题，选择带有记忆能力的神经网络效果会更好，本实验选择以GRU神经网络为基础进行实验。

## 介绍

### 1.数据集的准备

​	从Kaggle网站上下载已经标注好的数据集，“IMDB Dataset.csv”作为训练集，“TEST.csv”作为测试集
​	以上即为原始数据集，保存在与代码文件相同的目录下，以便后续调入使用

### 2.导入模块

​	此部分介绍使用的模块，并对其中比较重要的模块进行介绍：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import string
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import seaborn as sns
from tqdm import tqdm
import time
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchtext.vocab import Vectors
from torchtext.legacy import data
from sklearn.metrics import accuracy_score
```

​	在上面的库和模块中，nltk库对文本进行清洗与等预处理操作；torchtext库可以生成PyTorch框架下使用的文本张量；numpy与pandas为两个科学计算与数据分析的库，其中numpy包括了数组计算的功能，并且PyTorch计算所用张量与numpy中的数组转化十分方便；re库为正则表达式操作库，主要对文件中的文本正则化，方便后续词向量分析。

### 3.文本数据预处理

​	首先读入数据：

```python
# 获得训练集与测试集
train = pd.read_csv('IMDB Dataset.csv', sep=',', header=0, names=['sent', 'ment'])
test = pd.read_csv('Train.csv', header=0, names=['sent', 'ment'])
```

​	对于得到的数据，不能直接进行网络训练，需要对文本数据进行正则化与清洗：

```python
def get_label(la):
    label = []
    for s in la['ment']:
        if s == 'pos' or s == 'positive':
            label.append(1)
        elif s == 'neg' or s == 'negative':
            label.append(0)
    return label


def text_clean(text_data):
    text_pre = []
    for text1 in tqdm(text_data):
        text1 = re.sub("<br /><br />", " ", text1)
        text1 = text1.lower()
        text1 = re.sub("\d+", "", text1)
        text1 = text1.translate(
            str.maketrans("", "", string.punctuation.replace("'", ""))
        )
        text1 = text1.strip()
        text_pre.append(text1)
    return np.array(text_pre)


def cleaner(datalist, stop):
    data_pre = []
    for text in tqdm(datalist):
        # 分词
        text_words = word_tokenize(text)
        # 去除停用词
        text_words = [word for word in text_words if not word in stop]
        text_words = [word for word in text_words if len(re.findall("'", word)) == 0]
        data_pre.append(text_words)
    return np.array(data_pre)

# 获得标签
train_label = get_label(train)
test_label = get_label(test)
# 预处理文本
tran_c = text_clean(train['sent'])
test_c = text_clean(test['sent'])
# 去除停用词
stop = stopwords.words("english")
stop = set(stop)
tran_c2 = cleaner(tran_c, stop)
test_c2 = cleaner(test_c, stop)
```

​	首先获得数据集的标签，由于标注的是“positive”与“negative”，我们需将其转化为只包含0与1的数组备用。随后对于review部分，我们要将其正则化，利用text_clean函数，将所有的字母转化为小写字母，并去除数字、去除标点符号、去除多余的空格；仅经过正则化没有去除文本中的冗余信息（比如停用词），需利用cleaner函数进行进一步处理，其中，停用词可以通过stopwords.words("english")获得英文常用停用词。

​	完成数据准备后，将切分好的文本保存为数据表中的“text”变量，文本词之间用空格连接，文本的情感标签用“label”变量储存，将整理好的数据导入到csv文件后续调用：

```python
# 分好的数据存成csv备用
texts = [" ".join(words) for words in tran_c2]
traindatasave = pd.DataFrame({"text": texts,
                              "label": train_label})
texts = [" ".join(words) for words in test_c2]
testdatasave = pd.DataFrame({"text": texts,
                             "label": test_label})
traindatasave.to_csv("imdb_train.csv", index=False)
testdatasave.to_csv("imdb_test.csv", index=False)
```

​	结果如下：

![image-20210515184228337](C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20210515184228337.png)

![image-20210515184140609](C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20210515184140609.png)



### 4.分词与加权

​	数据经过预处理并存储以后，通过torchtext库将文件导入Python中，并处理为GRU神经网络能读取的数据结构。首先定义文件中对文本和标签所要做的操作，定义data.Field()实例来处理文本，对文本内容使用空格切分为词语：

```python
# 定义切分方法
myt = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=myt, include_lengths=True, use_vocab=True, batch_first=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

train_text_fields = [
    ("text", TEXT),
    ("label", LABEL)
]

# 读取数据
traindata, testdata = data.TabularDataset.splits(
    path="E:/USTC/secrity/AI/LAB5_NLP", format="csv", train="imdb_train.csv", fields=train_text_fields,
    test="imdb_test.csv", skip_header=True
)
```

​	定义好实例后，使用data.TabularDataset.splits()函数读取文本数据，以csv格式存储列的数据集，并按照实例定义的方式进行预处理。

​	读取文件以后，使用已经训练好的词向量构建词汇表，然后针对训练数据和测试数据使用data.BucketIterator()将其处理为数据加载器，每个batch包含32个文本数据：

```python
# 导入词向量文件 glove.6B.100d.txt
vec = Vectors("glove.6B.100d.txt", "./data")
TEXT.build_vocab(traindata, max_size=20000, vectors=vec)
LABEL.build_vocab(traindata)

tran_iter = data.BucketIterator(traindata, batch_size=32, device=device)
test_iter = data.BucketIterator(testdata, batch_size=32, device=device)
```

​	其中，为了加速训练速度，使用GPU训练网络，在程序的最开始需要定义使用的显卡：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')
```

​	然后在定义data.BucketIterator()时，使用device参数将数据定义为GPU可以读取的数据形式。最终得到训练数据与测试数据：tran_iter与test_iter。

### 5.建立网络与初始化

​	首先使用PyTorch定义一个GRUNet类：

```python
class GRUNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, layer_dim, output_dim):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        #GRU
        self.gru = nn.GRU(embed_dim, hidden_dim, layer_dim, batch_first=True)
        #全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        embeds = self.embedding(x)
        r_out, h_n = self.gru(embeds, None)
        out = self.fc1(r_out[:, -1, :])
        return out
```

​	对网络中的参数进行说明：

- vocab_size：词典长度
- embed_dim：词向量维度
- hidden_dim：GRU神经元个数
- layer_dim：GRU层数
- output_dim：隐藏层输出维度，即分类的个数

​     在GRU网络中，通过nn.GRU()定义对文本的循环层，然后通过全连接层进行分类处理，并且用torch.nn.Dropout(0.5)来减轻网络的过拟合。

​	将网络相关的参数进行初始化：

```python
vocab_size = len(TEXT.vocab)
embed_dim = vec.dim
hidden_dim = 128
layer_dim = 1
output_dim = 2

grumodel = GRUNet(vocab_size, embed_dim, hidden_dim, layer_dim, output_dim)
```

​	其中，hidden_dim就是神经元个数，output_dim就是分类个数，这里等于2，说明是个二分类问题。

### 6.定义优化器

​	接下来需要定义优化器optimizer，使用optim.RMSprop()类进行定义，损失函数使用交叉熵，其中交叉熵损失函数常用于二分类或者多分类问题。

​	RMSProp算法与动量梯度下降一样，都是消除梯度下降过程中的摆动来加速梯度下降的方法。 更新权重的时候，使用除根号的方法，整个梯度下降的过程中摆动就会比较小，就能设置较大的学习率，使得学习步频变大，达到加快学习的目的。

​	相关代码如下：

```python
# 定义优化器
optimizer = optim.RMSprop(grumodel.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()
```

### 7.训练网络与过程可视化

#### 7.1训练函数

​	定义一个训练网络的函数来对网络进行训练：

```python
def train_model(model, traindataloader, testdataloader, criterion, optimizer, num_epochs=25):
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    learn_rate = []
    since = time.time()
	#学习率调整
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(num_epochs):
        learn_rate.append(scheduler.get_last_lr()[0])
        print('-' * 10)
        print('Epoch {}/{},Lr:{}'.format(epoch, num_epochs - 1, learn_rate[-1]))
        train_loss = 0.0
        train_corrects = 0
        train_num = 0.0
        test_loss = 0.0
        test_corrects = 0
        test_num = 0
        model.train() #设置为训练模式
        for step, batch in enumerate(traindataloader):
            textdata, target = batch.text[0], batch.label
            out = model(textdata)
            pre_lab = torch.argmax(out, 1) #预测的label值
            loss = criterion(out, target) #计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
            train_corrects += torch.sum(pre_lab == target.data)
            train_num += len(target)
        #以下计算一次训练的损失与精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        print('{}:Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        scheduler.step() #更新学习率
        
        model.eval() #设置为评测模式
        for step, batch in enumerate(testdataloader):
            textdata, target = batch.text[0], batch.label
            out = model(textdata)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, target)
            test_loss += loss.item() * len(target)
            test_corrects += torch.sum(pre_lab == target.data)
            test_num += len(target)
        test_loss_all.append(test_loss / test_num)
        test_acc_all.append(test_corrects.double().item() / test_num)
        print('{}:test Loss:{:.4f} test Acc:{:.4f}'.format(epoch, test_loss_all[-1], test_acc_all[-1]))
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "test_loss_all": test_loss_all,
              "test_acc_all": test_acc_all})
    return model, train_process
```

​	下面对输入参数进行一个简要的说明：

- model：网络模型
- traindataloader：训练数据集
- testdataloader：测试数据集
- criterion：损失函数
- optimizer：优化方式
- num_epochs：训练轮数

​	下面是对学习率的动态调整函数：

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
```

​	设置的是等间隔调整学习率，每次缩小原来的0.1倍，每5轮缩小一次，其余见注释。

​	为了加快网络的训练速度，本次实验将使用已经训练好的词向量初始化词嵌入层的参数，将词向量作为embedding.weight的初始值，并将无法识别的词的向量初始化为0：

```python
grumodel.cuda()
grumodel.embedding.weight.data.copy_(TEXT.vocab.vectors)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
grumodel.embedding.weight.data[UNK_IDX] = torch.zeros(vec.dim)
grumodel.embedding.weight.data[PAD_IDX] = torch.zeros(vec.dim)
```

​	其中，由于使用GPU模式，在做这些操作之前，需要将模型转化为GPU形式，命令即grumodel.cuda()，保证输入、输出、模型均位于GPU。

​	此外，训练函数将将训练和测试两个阶段结合在一起，方便以后可视化，选择训练25轮：

```python

# 进行训练20轮
grumodel, train_process = train_model(
    grumodel, tran_iter, test_iter, loss_func, optimizer, num_epochs=25
)
# 可视化训练过程
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(train_process.epoch, train_process.train_loss_all, "r.-", label="Train Loss")
plt.plot(train_process.epoch, train_process.test_loss_all, "bs-", label="Test Loss")
plt.legend()
plt.xlabel("Epoch", size=13)
plt.ylabel("LOSS VALUE", size=13)

plt.subplot(1, 2, 2)
plt.plot(train_process.epoch, train_process.train_acc_all, "r.-", label="Train Acc")
plt.plot(train_process.epoch, train_process.test_acc_all, "bs-", label="Test Acc")
plt.xlabel("Epoch", size=13)
plt.ylabel("ACC VALUE", size=13)
plt.legend()
plt.show()
```

​	训练结束后会返回一个训练好的grumodel的模型，以及在训练过程中相关评价指标的集合train_process数据表。通过这个数据表即可完成可视化，得到的图像在实验结果中分析。

### 8.测试集精度

​	最后再输出一遍用训练好的模型预测测试集的精度，其中grumodel.cpu()是把网络从GPU转移到CPU以便后续的输出：

```python
# 测试集精度
grumodel.cpu()
test_iter = data.BucketIterator(testdata, batch_size=32)
grumodel.eval()
test_y_all = torch.LongTensor()
pre_lab_all = torch.LongTensor()
for step, batch in enumerate(test_iter):
    textdata, target = batch.text[0], batch.label.view(-1)
    out = grumodel(textdata)
    pre_lab = torch.argmax(out, 1)
    test_y_all = torch.cat((test_y_all, target))
    pre_lab_all = torch.cat((pre_lab_all, pre_lab))
acc = accuracy_score(test_y_all, pre_lab_all)
print("测试集精度：", acc)
```
