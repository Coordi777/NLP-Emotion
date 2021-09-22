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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')


# 网络模型
class GRUNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, layer_dim, output_dim):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, layer_dim, batch_first=True)
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


# 训练函数
def train_model(model, traindataloader, testdataloader, criterion, optimizer, num_epochs=25):
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    learn_rate = []
    since = time.time()

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
        model.train()
        for step, batch in enumerate(traindataloader):
            textdata, target = batch.text[0], batch.label
            out = model(textdata)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
            train_corrects += torch.sum(pre_lab == target.data)
            train_num += len(target)
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        print('{}:Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        scheduler.step()
        model.eval()
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


def get_label(la):
    label = []
    for s in la['ment']:
        if s == 'pos' or s == 1 or s == 'positive':
            label.append(1)
        elif s == 'neg' or s == 0 or s == 'negative':
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


# 获得训练集与测试集
train = pd.read_csv('IMDB Dataset.csv', encoding = "ISO-8859-1",sep=',', header=0, names=['sent', 'ment'])
test = pd.read_csv('TEST.csv', encoding = "ISO-8859-1",header=0, names=['sent', 'ment'])

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

# 分好的数据存成csv备用
texts = [" ".join(words) for words in tran_c2]
traindatasave = pd.DataFrame({"text": texts,
                              "label": train_label})
texts = [" ".join(words) for words in test_c2]
testdatasave = pd.DataFrame({"text": texts,
                             "label": test_label})
traindatasave.to_csv("imdb_train.csv", index=False)
testdatasave.to_csv("imdb_test.csv", index=False)

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

# 导入词向量文件 glove.6B.100d.txt
vec = Vectors("glove.6B.100d.txt", "./data")
TEXT.build_vocab(traindata, max_size=20000, vectors=vec)
LABEL.build_vocab(traindata)

tran_iter = data.BucketIterator(traindata, batch_size=32, device=device)
test_iter = data.BucketIterator(testdata, batch_size=32, device=device)

# 初始化网络
vocab_size = len(TEXT.vocab)
embed_dim = vec.dim
hidden_dim = 128
layer_dim = 1
output_dim = 2

grumodel = GRUNet(vocab_size, embed_dim, hidden_dim, layer_dim, output_dim)
grumodel.cuda()
grumodel.embedding.weight.data.copy_(TEXT.vocab.vectors)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
grumodel.embedding.weight.data[UNK_IDX] = torch.zeros(vec.dim)
grumodel.embedding.weight.data[PAD_IDX] = torch.zeros(vec.dim)

# 定义优化器
optimizer = optim.RMSprop(grumodel.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()
# 进行训练 10轮
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
