import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn


def compute_accuracy(net, dataset, **kwargs):
	"""
	Подчсет точности классификации
	"""
    correct=0
    total=0
    for i, data in enumerate(dataset):
        images, labels = data
        if (kwargs):
            outputs = net.forward(Variable(images), kwargs)
        else:
            outputs = net.forward(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (labels == predicted).sum()

    print('Accuracy on dataset: {:.3f}'.format(correct / total))
    return correct / total

def optimize_net(net, criterion, optimizer, dataset, num_epochs=1, show_progress=True, **kwargs):
   	"""
   	Оптимизация нейросети
   	"""
    for epoch in range(num_epochs):
    
        running_loss = 0;
        for i, data in enumerate(dataset):
            imgs, labels = data
            imgs, labels = Variable(imgs), Variable(labels)
            optimizer.zero_grad()
            if (kwargs):
                outputs = net.forward(imgs, kwargs)
            else:
                outputs = net.forward(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.data[0]
            if ((i + 1) % 500 == 0) and (show_progress is True):
                print("[{:d}, {:5d}] loss: {:.3f}".format(epoch + 1, i + 1, running_loss / 500))
                running_loss = 0

def optimize_encoder(net, criterion, optimizer, dataset, num_epochs=1, show_progress=True):
	"""
	Оптимизация автокодировщика. 
	Отличается от функции для оптимизации нейросети тем, что оптимизируется тождественная функция.
	"""
    for epoch in range(num_epochs):
    
        running_loss = 0;
        for i, data in enumerate(dataset):
            imgs, _ = data
            imgs = Variable(imgs)
            optimizer.zero_grad()
            outputs = net.forward(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.data[0]
            if ((i + 1) % 2000 == 0) and (show_progress is True):
                print("[{:d}, {:5d}] loss: {:.3f}".format(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0

class ConvAutoEncoder(nn.Module):
	"""
	Реализация автокодировщика и сверточной нейросети с произвольным количеством сверточных блоков. 
	Способы прохода по графу вычислений задаются флагами в методе forward.
	В качестве нелинейного преобразования на последнем слое декодера используется функция tanh,
	областью значений которой является отрезок  [−1,1], так 
	как значения пикселей нормализованного изображения лежат в этом отрезке.
	"""
    def __init__(self, input_size=(3, 32, 32), layers_num=2, conv_out_channels=(6, 6), conv_kernel_size=(4, 4),
                 conv_stride=(2, 1), pool_kernel_size=(2, 1), pool_stride=(1, 1), conv_padding=(0, 0),
                 linear_layers_num=2, linear_out_channels=(84, 10)):
        super(ConvAutoEncoder, self).__init__()
        self.input_size = input_size
        self.layers_num = layers_num
        self.linear_layers_num = linear_layers_num
        conv_in_channels = [input_size[0]]
        conv_in_channels.extend(conv_out_channels[:-1:])
        
        self.conv = nn.ModuleList([nn.Conv2d(conv_in_channels[i], conv_out_channels[i], 
                                  kernel_size=conv_kernel_size[i], stride=conv_stride[i], padding=conv_padding[i])
                                  for i in range(layers_num)])
        self.pool = nn.ModuleList([nn.MaxPool2d(kernel_size=pool_kernel_size[i],
                                  stride=pool_stride[i], return_indices=True) for i in range(layers_num)])
        self.unpool = nn.ModuleList([nn.MaxUnpool2d(kernel_size=pool_kernel_size[- 1 - i],
                                    stride=pool_stride[-1 - i]) for i in range(layers_num)])
        self.convTrans = nn.ModuleList([nn.ConvTranspose2d(conv_out_channels[-1 - i], 
                                       conv_in_channels[-1 -i], kernel_size=conv_kernel_size[-1 - i],
                                       stride=conv_stride[-1 - i], padding=conv_padding[-1 - i]) 
                                       for i in range(layers_num)])
       
        output_size_x = input_size[1]
        output_size_y = input_size[2]
        
        for i in range(layers_num):
            output_size_x = math.floor((output_size_x + 2 * conv_padding[i]  - conv_kernel_size[i]) / conv_stride[i] + 1)
            output_size_y = math.floor((output_size_y + 2 * conv_padding[i] - conv_kernel_size[i]) / conv_stride[i] + 1)
            output_size_x = math.floor((output_size_x - pool_kernel_size[i]) / pool_stride[i] + 1)
            output_size_y = math.floor((output_size_y - pool_kernel_size[i]) / pool_stride[i] + 1)
        
        linear_in_channels = [conv_out_channels[-1] * output_size_x * output_size_y]
        linear_in_channels.extend(linear_out_channels[:-1:])
        self.linear = nn.ModuleList([nn.Linear(linear_in_channels[i], linear_out_channels[i]) for i in range(linear_layers_num)])

    def forward(self, x, AutoEncoder=True, TransformObjects=False):
        indices = []
        output_size = []
        for i in range(self.layers_num):
            x = F.relu(self.conv[i](x))
            output_size.append(x.size())
            x, ind = self.pool[i](x)
            indices.append(ind)
        if (TransformObjects is True):
            return x.view(-1, self.num_features(x))
        if (AutoEncoder is True):
            for i in range(self.layers_num):
                x = self.unpool[i](x, indices[-1 - i], output_size=output_size[-1 - i])
                if i == self.layers_num - 1:
                    x = F.tanh(self.convTrans[i](x))
                else:
                    x = F.relu(self.convTrans[i](x))
        else:
            x = x.view(-1, self.num_features(x))
            for i in range(self.linear_layers_num - 1):
                x = F.relu(self.linear[i](x))
            x = self.linear[-1](x)
        return x
    
    def num_features(self, x):
        return int(np.array(x.size()[1::]).prod())