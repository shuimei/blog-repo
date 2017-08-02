---
title: CNN(深度神经网络)——基于PyTorch，以MNIST手写数据识别为例
---

部分引自[莫烦PyTorch教程][1]

PyTorch 是 Torch 在 Python 上的衍生. 因为 Torch 是一个使用 Lua 语言的神经网络库, Torch 很好用, 但是 Lua 又不是特别流行, 所有开发团队将 Lua 的 Torch 移植到了更流行的语言 Python 上. 是的 PyTorch 一出生就引来了剧烈的反响. 为什么呢?
很简单, 我们就看看有谁在用 PyTorch 吧.
![此处输入图片的描述][2]
可见, 著名的 Facebook, twitter 等都在使用它, 这就说明 PyTorch 的确是好用的, 而且是值得推广.
而且如果你知道 Numpy, PyTorch 说他就是在神经网络领域可以用来替换 numpy 的模块.
据 PyTorch 自己介绍, 他们家的最大优点就是建立的神经网络是动态的, 对比静态的 Tensorflow, 他能更有效地处理一些问题, 比如说 RNN 变化时间长度的输出. 而我认为, 各家有各家的优势和劣势, 所以我们要以中立的态度. 两者都是大公司, Tensorflow 自己说自己在分布式训练上下了很大的功夫, 那我就默认 Tensorflow 在这一点上要超出 PyTorch, 但是 Tensorflow 的静态计算图使得他在 RNN 上有一点点被动 (虽然它用其他途径解决了), 不过用 PyTorch 的时候, 你会对这种动态的 RNN 有更好的理解.

而且 Tensorflow 的高度工业化, 它的底层代码… 你是看不懂的. PyTorch 好那么一点点, 如果你深入 API, 你至少能比看 Tensorflow 多看懂一点点 PyTorch 的底层在干嘛.

最后我的建议就是:

如果你是学生, 随便选一个学, 或者稍稍偏向 PyTorch, 因为写代码的时候应该更好理解. 懂了一个模块, 转换 Tensorflow 或者其他的模块都好说.
如果是上班了, 跟着你公司来, 公司用什么, 你就用什么, 不要脱群.

## 安装
目前PyTorch仅支持Linux和OSX平台，如果你想体验一下PyTorch，可以安装一个Ubuntu虚拟机。
安装PyTorch与安装其它Python模块非常相似。首先访问[Pytorch官网][3]，选择相应选项：
![](/home/betasy/Desktop/深度截图_选择区域_20170710161855.png)

得到如下的安装命令，使用pip安装。
```
pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl 
pip install torchvision
```
安装结束后在命令行中运行`import torch;import torchvision`测试安装是否成功。
安装 PyTorch 会安装两个模块, 一个是 torch, 一个 torchvision, torch 是主模块, 用来搭建神经网络的, torchvision 是辅模块, 有数据库, 还有一些已经训练好的神经网络等着你直接用, 比如 (VGG, AlexNet, ResNet).

## Tensor和Variable

+ **Tensor**
Torch 自称为神经网络界的 Numpy, 因为他能将 torch 产生的 tensor 放在 GPU 中加速运算 (前提是你有合适的 GPU), 就像 Numpy 会把 array 放在 CPU 中加速运算. 所以神经网络的话, 当然是用 Torch 的 tensor 形式数据最好咯. 就像 Tensorflow 当中的 tensor 一样.
当然, 我们对 Numpy 还是爱不释手的, 因为我们太习惯 numpy 的形式了. 不过 torch 看出来我们的喜爱, 他把 torch 做的和 numpy 能很好的兼容. 比如这样就能自由地转换 numpy array 和 torch tensor 了:
```
import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,          # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array, # [[0 1 2], [3 4 5]]
)
```
其实 torch 中 tensor 的运算和 numpy array 的如出一辙, 我们就以对比的形式来看. 如果想了解 torch 中其它更多有用的运算符, [API就是你要去的地方](http://pytorch.org/docs/torch.html#math-operations) .
```
# abs 绝对值计算
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),          # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor)      # [1 2 1 2]
)

# sin   三角函数 sin
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

# mean  均值
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),         # 0.0
    '\ntorch: ', torch.mean(tensor)     # 0.0
)
```
除了简单的计算, 矩阵运算才是神经网络中最重要的部分. 所以我们展示下矩阵的乘法. 注意一下包含了一个 numpy 中可行, 但是 torch 中不可行的方式.
```
# matrix multiplication 矩阵点乘
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
)

# !!!!  下面是错误的方法 !!!!
data = np.array(data)
print(
    '\nmatrix multiplication (dot)',
    '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]] 在numpy 中可行
    '\ntorch: ', tensor.dot(tensor)     # torch 会转换成 [1,2,3,4].dot([1,2,3,4) = 30.0
)
```
+ **Variable**
在 Torch 中的 Variable 就是一个存放会变化的值的地理位置. 里面的值会不停的变化. 就像一个裝鸡蛋的篮子, 鸡蛋数会不停变动. 那谁是里面的鸡蛋呢, 自然就是 Torch 的 Tensor 咯. 如果用一个 Variable 进行计算, 那返回的也是一个同类型的 Variable.

我们定义一个 Variable:
```
import torch
from torch.autograd import Variable # torch 中 Variable 模块

# 先生鸡蛋
tensor = torch.FloatTensor([[1,2],[3,4]])
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)

print(tensor)
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable)
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""
```

我们再对比一下 tensor 的计算和 variable 的计算.
```
t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
print(t_out)
print(v_out)    # 7.5
到目前为止, 我们看不出什么不同, 但是时刻记住, Variable 计算时, 它在背景幕布后面一步步默默地搭建着一个庞大的系统, 叫做计算图, computational graph. 这个图是用来干嘛的? 原来是将所有的计算步骤 (节点) 都连接起来, 最后进行误差反向传递的时候, 一次性将所有 variable 里面的修改幅度 (梯度) 都计算出来, 而 tensor 就没有这个能力啦.

v_out = torch.mean(variable*variable) 就是在计算图中添加的一个计算步骤, 计算误差反向传递的时候有他一份功劳, 我们就来举个例子:

v_out.backward()    # 模拟 v_out 的误差反向传递
```
下面两步看不懂没关系, 只要知道 Variable 是计算图的一部分, 可以用来传递误差就好.
 v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤
 针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4*2*variable = variable/2
```
print(variable.grad)    # 初始 Variable 的梯度
'''
 0.5000  1.0000
 1.5000  2.0000
'''
```
直接print(variable)只会输出 Variable 形式的数据, 在很多时候是用不了的(比如想要用 plt 画图), 所以我们要转换一下, 将它变成 tensor 形式.
```
print(variable)     #  Variable 形式
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)    # tensor 形式
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())    # numpy 形式
"""
[[ 1.  2.]
 [ 3.  4.]]
"""
```
## 激励函数(Activation)
一句话概括 Activation: 就是让神经网络可以描述非线性问题的步骤, 是神经网络变得更强大. 
Torch 中的激励函数有很多, 不过我们平时要用到的就这几个. relu, sigmoid, tanh, softplus. 那我们就看看他们各自长什么样啦.
```
import torch
import torch.nn.functional as F     # 激励函数都在这
from torch.autograd import Variable

# 做一些假数据来观看图像
x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
x = Variable(x)
接着就是做生成不同的激励函数数据:

x_np = x.data.numpy()   # 换成 numpy array, 出图时用

# 几种常用的 激励函数
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
# y_softmax = F.softmax(x)  softmax 比较特殊, 不能直接显示, 不过他是关于概率的, 用于分类
```
![](/home/betasy/Documents/PyTorch-Tutorial/2-3-1.png) 
```
import matplotlib.pyplot as plt  # python 的可视化模块, 我有教程 (https://morvanzhou.github.io/tutorials/data-manipulation/plt/)

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
```
## 关系拟合（回归）
我们创建一些假数据来模拟真实的情况. 比如一个一元二次函数: y = a * x^2 + b, 我们给 y 数据加上一点噪声来更加真实的展示它.
```
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# 用 Variable 来修饰这些数据 tensor
x, y = torch.autograd.Variable(x), Variable(y)

# 画图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
```
建立一个神经网络我们可以直接运用 torch 中的体系. 先定义所有的层属性(__init__()), 然后再一层层搭建(forward(x))层于层的关系链接. 建立关系的时候, 我们会用到激励函数, 如果还不清楚激励函数用途的同学, 这里有非常好的一篇动画教程.
```
import torch
import torch.nn.functional as F     # 激励函数都在这

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  # net 的结构
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""
```

训练的步骤很简单, 如下:

```
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
for t in range(100):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
```
为了可视化整个训练的过程, 更好的理解是如何训练, 我们如下操作:
```
import matplotlib.pyplot as plt

plt.ion()   # 画图
plt.show()

for t in range(100):

    ...
    loss.backward()
    optimizer.step()

    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
```
![](/home/betasy/Documents/PyTorch-Tutorial/3-1-1.png) 
## 区分类型（分类）
我们创建一些假数据来模拟真实的情况. 比如两个二次分布的数据, 不过他们的均值都不一样.
```
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

# torch 只能在 Variable 上训练, 所以把它们变成 Variable
x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# 画图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
```
建立一个神经网络我们可以直接运用 torch 中的体系. 先定义所有的层属性(__init__()), 然后再一层层搭建(forward(x))层于层的关系链接. 这个和我们在前面 regression 的时候的神经网络基本没差. 建立关系的时候, 我们会用到激励函数, 如果还不清楚激励函数用途的同学, 这里有非常好的一篇动画教程.
```
import torch
import torch.nn.functional as F     # 激励函数都在这

class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2) # 几个类别就几个 output

print(net)  # net 的结构
"""
Net (
  (hidden): Linear (2 -> 10)
  (out): Linear (10 -> 2)
)
"""
```

训练的步骤很简单, 如下:
```
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)     # 喂给 net 训练数据 x, 输出分析值

    loss = loss_func(out, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
```
为了可视化整个训练的过程, 更好的理解是如何训练, 我们如下操作:
```
import matplotlib.pyplot as plt

plt.ion()   # 画图
plt.show()

for t in range(100):

    ...
    loss.backward()
    optimizer.step()

    # 接着上面来
    if t % 2 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()
```
![](/home/betasy/Documents/PyTorch-Tutorial/3-2-1.png) 
## 手写数字识别（CNN卷积神经网络）
我从kaggle竞赛官网上下载了MNIST手写数据集
[https://www.kaggle.com/c/digit-recognizer](https://www.kaggle.com/c/digit-recognizer) 
这个竞赛的手写数据是以csv格式存储的，我首先将其转换成了图片。每个图片都是一张28*28的灰度图像，表示一个手写的数字。
![](https://kaggle2.blob.core.windows.net/competitions/kaggle/3004/logos/front_page.png) 

1. **创建数据集**

手写数据集需要使用`torch.utils.data`模块包装为torch可处理的标准格式。通过继承该模块的Dataset类，并重写`__init__`和`__getitem__`方法，返回相应的训练数据和标签的迭代器。
```
class MNISTDataset(Data.Dataset):
    """docstring for MyDataset"""

    def __init__(self, images, labels):
        super(MNISTDataset, self).__init__()
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        img_obj = Image.open(img)
        arr = np.asarray(img_obj, dtype="float32")
        arr = arr.reshape((1, arr.shape[0], arr.shape[1]))
        img_tensor = torch.from_numpy(arr)
        target_tensor = torch.LongTensor(np.array([int(target)]))
        return img_tensor, target_tensor

    def __len__(self):
        return len(self.images)
```
2. **构建神经网络**
PyTorch构建神经网络非常简单，这里只需要继承`torch.nn`中的`Module`就可以使用一系列的基础设施建设我们的神经网络了。这里实现了一个简单的卷积神经网络，它包括`conv1`,`conv2`和`out`层，其中`conv1`,`conv2`是两个卷积层，`out`是一个类似全连接层的结构。
除了设计网路结构，还要定义层与层之间的前向传播规则。定义`forward`方法可以实现前向传播的规则。
![](http://b.hiphotos.baidu.com/baike/w%3D268%3Bg%3D0/sign=2d6e940a38292df597c3ab13840a3b5d/b999a9014c086e06a4156e6e00087bf40ad1cb52.jpg) 
我们可以把`__init__`方法看做是上图中的神经元，而`forward`方法就是神经元之间的连线
```
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
```
## 优化网络
创建好网络结构之后，我们可以预见网络中会有许多参数（weights, bias）需要求解。卷积神经网络运行的过程，就是优化器按照一定的规则对网络中的参数进行调整的过程。PyTorch提供几种常用的优化器。只需要按照需求实例化优化器就好了。
`optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)`
## 损失函数
损失函数是优化器依赖的条件，同样的Pytorch中提供多种损失函数应对不同的计算任务。
`loss_func = nn.CrossEntropyLoss()`
## 训练
```
plt.ion()
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            b_y = b_y.view(-1)
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 ==0:
            	print('Epoch: ', epoch, "| train loss: %.4f" % loss.data[0])
            	plt.plot(step, loss.data[0],marker="o",markeredgecolor='red', markersize=8)
            	plt.show(); plt.pause(0.01)
    plt.ioff()
    ```
    ## 保存模型
    `torch.save(cnn.state_dict(), "mnist_cnn_params_epoch2.pkl")`
    ## 测试
    ```
    cnn2= CNN()
cnn2.load_state_dict(torch.load("mnist_cnn_params_epoch2.pkl"))
f = open("submission2.csv","a+")
f.write("ImageId,Label\n")
for i in range(1,28001):
    img_path = "/home/betasy/Documents/kaggle/digits recognizer/mnist/test/%d.jpg"%i
    img_obj = Image.open(img_path)
    arr = np.asarray(img_obj, dtype="float32")

    arr = arr.reshape((1, 1, arr.shape[0], arr.shape[1]))
    
    img_tensor = torch.from_numpy(arr)
    x = Variable(img_tensor)
    output = cnn2(x)
    pred_y = torch.max(output, 1)[1].data.squeeze()
    prediction = pred_y.numpy()
    f.write("%d,%d\n"%(i, int(prediction)))
f.close()
print("prediction over.")
```
  [1]: https://morvanzhou.github.io/tutorials/machine-learning/torch/
  [2]: https://morvanzhou.github.io/static/results/torch/1-1-1.png
  [3]: http://pytorch.org/