---
title: 用PyTorch做深度学习之fine-tuning
---
### 问题描述
问题来自kaggle竞赛网站上的一个比赛：[入侵物种监测](https://www.kaggle.com/c/invasive-species-monitoring) 。在这个比赛中，挑战者需要对人为拍摄的图片进行处理，分析其中是否包含一种绣球花（入侵物种）。
<center>有绣球花：</center>
<center><img src="train/3.jpg" width = "300" height = "200" /> 
</center>
<center>没有绣球花：</center>
 <center><img src="./train/1.jpg" width = "300" height = "200" /> </center>
上面是图片样例。

我发现这个题目是一个典型的二分类题目， 有绣球花的图片与没有绣球花的图片呈现出非常大的差异。可以通过训练一个卷积神经网络提取图片的主要特征用于识别。但是要注意的是，本例中图片大小为1154x866。已经做过一些尝试，搭建两个隐层的神经网络进行训练的运算开销对于单机CPU来说就已经吃不消。于是我想到了使用现有模型进行fine-tuning的方案。

### 导入模型
PyTorch支持从某个地址导入已有模型。相关模块在`torchvision.models`中。
预存的模型主要包括：

+ [AlexNet](https://arxiv.org/abs/1404.5997) 
+ [ VGG](https://arxiv.org/abs/1409.1556) 
+ [ResNet](https://arxiv.org/abs/1512.03385) 
+ [SqueezeNet](https://arxiv.org/abs/1602.07360) 
+ [DenseNet](https://arxiv.org/abs/1608.06993) 

如果要加载模型，并且随机初始化权重，可以：
```
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
squeezenet = models.squeezenet1_0()
densenet = models.densenet_161()
```
但是我们希望可以使用已经训练好的参数，则可以加上`pretrained`参数，加载训练好的模型的权重。
```
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
```
AlexNet的结构如下所示
```
AlexNet (
  (features): Sequential (
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU (inplace)
    (2): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1))
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU (inplace)
    (5): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1))
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU (inplace)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU (inplace)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU (inplace)
    (12): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1))
  )
  (classifier): Sequential (
    (0): Dropout (p = 0.5)
    (1): Linear (9216 -> 4096)
    (2): ReLU (inplace)
    (3): Dropout (p = 0.5)
    (4): Linear (4096 -> 4096)
    (5): ReLU (inplace)
    (6): Linear (4096 -> 1000)
  )
)
```
在PyTorch中，所有的预训练模型都需要输入经过相同方式归一化的图像，（3×H×W），H和W至少为224。在输入图像时需要注意。
加载AlexNet，如果要对网络结构进行修改，可以新建一个网络类，然后将AlexNet封装到新类中。
```
class AlexNetTransferModel(nn.Module):
	def __init__(self):
		super(AlexNetTransferModel, self).__init__()
		alexnet = models.alexnet(pretrained=True)
		for param in alexnet.parameters():
			param.requires_grad = False
		self.pretrained_model = alexnet
		self.last_layer = nn.Linear(1000, 2)

	def forward(self, x):
		return self.last_layer(self.pretrained_model(x))
```
如上所示，可以在新的网络结构类中添加加载预训练模型的代码到`__init__()`方法中。这里我在AlexNet的全连接层之后加了一层输入为1000，输出为2的全连接层，以便满足我当前问题的要求。要注意的是，如果我们采用了预训练的模型，且不再更新预训练模型的参数，需要对网络的权重进行固定，即遍历参数，并设置`requires_grad`为True。
这样就可以只更新最后一个全连接层的权重来调整模型了。
### 开始训练
```
import torchvision.models as models
from utilities import *
import torch.nn as nn

EPOCH = 2
BATCH_SIZE = 50
LR = 0.01

input_dir = "./train"
images_list, labels_list = read_images_list(input_dir, "train_labels.csv")
train_data = ISMDataset(images_list, labels_list)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)



model = AlexNetTransferModel()
optimizer = torch.optim.Adam(model.last_layer.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()
plt.ion()
loss_list = []
for epoch in range(EPOCH):
	for step, (x, y) in enumerate(train_loader):
		b_x = Variable(x)
		b_y = Variable(y)
		b_y = b_y.view(-1)
		output = model(b_x)
		# print(output.data)
		loss = loss_func(output, b_y)
		loss_list.extend(loss.data.numpy())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if step % 2 ==0:
			print('Epoch: ', epoch, "| train loss: %.4f" % loss.data[0])
			# plt.plot(step, loss.data[0],marker="o",markeredgecolor='red', markersize=4)
			plt.plot(loss_list, color="green", linestyle="dashed", marker="o", markeredgecolor='red', markersize=4)
			plt.show(); plt.pause(0.01)
torch.save(model.state_dict(),"cnn_alexnet_model_save_4.pkl")
plt.savefig("cnn_alexnet_model_save_4.jpg")
plt.ioff()
```
我将创建好的网络结构类同意存放在一个`utilities.py`文件中，直接导入就可以使用。下面是损失的变化情况：
<center>![](cnn_alexnet_model_save_4.jpg)</center>
 可以看出模型损失量在波动下降并逐渐收敛。
### 进行预测
在上一节中，我将训练好的模型存放在了`cnn_alexnet_model_save_4.pkl`文件中，可以直接加载模型进行预测。
```
from utilities import *
import os
test_model = AlexNetTransferModel()
test_model.load_state_dict(torch.load("cnn_alexnet_model_save_4.pkl"))

img_list = list(map(lambda a: "./test/%s"%a, os.listdir("./test")))
with open("test.csv", "a+") as f:
	f.write("name,invasive\n")
	for img_path in img_list:
		img = Image.open(img_path)
		img_tensor = img2tensor(img)
		var = Variable(img_tensor)
		output = test_model(var)
		# print(output)
		prob = F.softmax(output)
		p = torch.max(prob).data.numpy()
		name = img_path.split("/")[2][:-4]
		f.write("%s,%.4f\n" % (name, p))
		print("%s,%.4f" % (name, p))
```
提交结果到kaggle网站上，得到0.51862分
<center><img src="score_kaggle.png" width="800" height="800"/></center>

