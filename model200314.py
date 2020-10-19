#from wban import wbal
from collections import defaultdict
from utils import *
import torch.nn.functional as F
from torch import nn, tanh, sigmoid  , cat
import torch
from enhance0314 import Dehaze,Dehaze1
import PIL.Image as image
from matplotlib import  pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs

#meta = torch.load("./model/dehaze_80.pth",map_location="cuda:0")#,map_location="cuda:0"
meta = torch.load("./meta/dehaze_80.pth",map_location="cuda:0")
class FEA(nn.Module):

    def __init__(self):
        super(FEA, self).__init__()

    def forward(self, x):
        outs = []
        fea1, fea2, fea3, fea4, fea5, fea6 = meta(x)  # vgg16(pre_haze.expand(1, 3, 480, 640))
        outs.append(fea1)
        outs.append(fea2)
        outs.append(fea3)
        outs.append(fea4)
        outs.append(fea5)
        return fea1, fea2, fea3, fea4, fea5, fea6, outs

class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding = 1):
        super(MyConv2D, self).__init__()
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size)).to(device)
        self.bias = torch.zeros(out_channels).to(device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding,padding)

    def forward(self, x):

        return F.conv2d(x, self.weight, self.bias, self.stride,self.padding)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        return s.format(**self.__dict__)


class dMyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=0,output_padding = 0): #,output_padding = 1
        super(dMyConv2D, self).__init__()
        self.weight = torch.zeros((in_channels,out_channels,  kernel_size, kernel_size)).to(device)
        self.bias = torch.zeros(out_channels).to(device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.out_padding = (output_padding, output_padding)

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,self.out_padding)#,self.out_padding

        #return F.conv_transpose2d(
        #input, self.weight, self.bias, self.stride, self.padding,
        #output_padding, self.groups, self.dilation)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        return s.format(**self.__dict__)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1,padding = 1),
            *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False,padding = 1)
        )
        #self.sa = SpatialAttention(3)
        #self.ca = ChannelAttention(16*4)

    def forward(self, x):

        return self.conv(x) + x


def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,
              upsample=None, instance_norm=True, relu=True, trainable=False,padding = 1):
    layers = []

    if trainable:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding ))
    else:
        layers.append(MyConv2D(in_channels, out_channels, kernel_size, stride,padding))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU())
    return layers

class OALayer(nn.Module):
    def __init__(self, num_ops=34,channel=160,k=1):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
                    nn.Linear(channel, self.output*2),
                    nn.ReLU(),
                    nn.Linear(self.output*2, self.k*self.num_ops))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y



class TransformNet(nn.Module):
    def __init__(self, base=8):  # 初始化,传入相应的参数
        super(TransformNet, self).__init__()  # 对继承自父类的属性进行初始化
        self.base = base
        self.weights = []
        ##torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。
        self.net= nn.Sequential(
            *ConvLayer(12, base, kernel_size=3, stride=1, padding=1, trainable=True),  # stride=2
            #*ConvLayer(3, base , kernel_size=3, stride=1, padding=1,trainable=True),  # stride=2
            *ConvLayer(base, base * 2, kernel_size=3, stride=1, padding=1,trainable=True),  # stride=2
            *ConvLayer(base * 2,  base* 4, kernel_size=3, stride=1, padding=1,trainable=True),  # instance_norm=False, relu=False,

        )

        self.op11 = nn.Sequential(
            *ConvLayer(base * 4, base, kernel_size=1, stride=1, padding=0,trainable=True),  # trainable=True,
        )
        self.op22 = nn.Sequential(
            *ConvLayer(base* 4, base, kernel_size=3, stride=1, padding=1,trainable=True),  # trainable=True,
        )
        self.op33 = nn.Sequential(
            *ConvLayer(base * 4, base, kernel_size=5, stride=1, padding=2,trainable=True),  # stride=2
        )
        self.op44 = nn.Sequential(
            *ConvLayer(base * 4, base, kernel_size=7, stride=1, padding=3,trainable=True),  # trainable=True,
        )

        self.residuals = nn.Sequential(*[ResidualBlock(base * 4) ])
        #self.residuals3 = nn.Sequential(*[ResidualBlock(base * 4) for i in range(3)])
        self.netd = nn.Sequential(

            *ConvLayer(base *4, base * 2, kernel_size=3, stride=1, padding=1,trainable=True),  # stride=2
            *ConvLayer(base * 2, base, kernel_size=3, stride=1, padding=1,trainable=True),  # stride=2
            *ConvLayer(base, 3, kernel_size=3, stride=1, padding=1,trainable=True),#instance_norm=False, relu=False,

        )
        #self.rdb_in = RDB(3,3, 16)


        self.get_param_dict()

        self.dehaze = Dehaze(channles=3)
        self.dehaze1 = Dehaze1(channles=3)
        self.attention = OALayer()

    def forward(self, X,fea,pre):
        weights = F.softmax(self.attention(fea),dim = -1)


        x = self.net(pre)#x = self.net(X)
        x = self.residuals(x)

        y1 = self.op11(x) * weights[:, :, 0].view([-1, 1, 1, 1])
        y2 = self.op22(x) * weights[:, :, 1].view([-1, 1, 1, 1])
        y3 = self.op33(x) * weights[:, :, 2].view([-1, 1, 1, 1])
        y4 = self.op44(x) * weights[:, :, 3].view([-1, 1, 1, 1])

        y = cat([y1, y2, y3, y4], 1)
        #y = self.residuals(y)


        y11 = self.op11(y) * weights[:, :, 4].view([-1, 1, 1, 1])
        y22 = self.op22(y) * weights[:, :, 5].view([-1, 1, 1, 1])
        y33 = self.op33(y) * weights[:, :, 6].view([-1, 1, 1, 1])
        y44 = self.op44(y) * weights[:, :, 7].view([-1, 1, 1, 1])

        yy = cat([y11, y22, y33, y44], 1)+x
        #yy = self.residuals(yy)
        #yyb = self.con(yy)

        y111 = self.op11(yy) * weights[:, :, 8].view([-1, 1, 1, 1])
        y222 = self.op22(yy) * weights[:, :, 9].view([-1, 1, 1, 1])
        y333 = self.op33(yy) * weights[:, :, 10].view([-1, 1, 1, 1])
        y444 = self.op44(yy) * weights[:, :, 11].view([-1, 1, 1, 1])

        yyy = cat([y111, y222, y333, y444], 1)+y
        #yyy = self.residuals(yyy)
        #yyyb= self.con(yyy)

        y1111 = self.op11(yyy) * weights[:, :, 12].view([-1, 1, 1, 1])
        y2222 = self.op22(yyy) * weights[:, :, 13].view([-1, 1, 1, 1])
        y3333 = self.op33(yyy) * weights[:, :, 14].view([-1, 1, 1, 1])
        y4444 = self.op44(yyy) * weights[:, :, 15].view([-1, 1, 1, 1])

        yyyy = cat([y1111, y2222, y3333, y4444], 1)+yy
        #yyyy = self.residuals(yyyy)
        #yyyyb = self.con(yyyy)

        y11111 = self.op11(yyyy) * weights[:, :, 16].view([-1, 1, 1, 1])
        y22222 = self.op22(yyyy) * weights[:, :, 17].view([-1, 1, 1, 1])
        y33333 = self.op33(yyyy) * weights[:, :, 18].view([-1, 1, 1, 1])
        y44444 = self.op44(yyyy) * weights[:, :, 19].view([-1, 1, 1, 1])

        yyyyy = cat([y11111, y22222, y33333, y44444], 1) + yyy
        #yyyyy = self.residuals(yyyyy)

        y111111 = self.op11(yyyyy) * weights[:, :, 20].view([-1, 1, 1, 1])
        y222222 = self.op22(yyyyy) * weights[:, :, 21].view([-1, 1, 1, 1])
        y333333 = self.op33(yyyyy) * weights[:, :, 22].view([-1, 1, 1, 1])
        y444444 = self.op44(yyyyy) * weights[:, :, 23].view([-1, 1, 1, 1])

        yyyyyy = cat([y111111, y222222, y333333,y444444], 1) + yyyy
        #yyyyyy = self.residuals(yyyyyy)



       # ye = (cat((yb,yyb,yyyb,yyyyb),1))+yyyy
        #ye = self.residuals(yyyyyy)

        yed = self.netd(yyyyyy)+X

        yede = self.dehaze(yed,weights)+X

        dehazed = self.dehaze1(yede,weights) + X
        #weights_t = torch.rand(1, 1, 34)
        '''
        
        weights_t = weights[0, :, :].cpu()
        #print(weights_t)

        weights_n = weights_t.detach().numpy()
        #print(weights_n)
        weights_n1 = weights_n / weights_n.max(axis=1) * 255  # /weights_n.max(axis=1)
        # print(weights_n.size())
        new_im = image.fromarray(weights_n1.astype('uint8'),'L')
        new_im.save('./weights.bmp')
        im = image.open('./weights.bmp')
        plt.subplot(111)
        plt.imshow(im, cmap='Greys_r')
        plt.title('Visualization')
        plt.show()
        '''



        return dehazed



    def get_param_dict(self):
        """找出该网络所有 MyConv2D 层，计算它们需要的权值数量"""
        param_dict = defaultdict(int)  # defaultdict(int)：初始化为 0

        def dfs(module, name):
            for name2, layer in module.named_children():
                dfs(layer, '%s.%s' % (name, name2) if name != '' else name2)
            if module.__class__ == MyConv2D:
                param_dict[name] += int(np.prod(module.weight.shape))  # np.prod()函数用来计算所有元素的乘积
                param_dict[name] += int(np.prod(module.bias.shape))

        dfs(self, '')  # self指的是类实例对象本身
        return param_dict

    def set_my_attr(self, name, value):
        # 下面这个循环是一步步遍历类似 residuals.0.conv.1 的字符串，找到相应的权值
        target = self
        for x in name.split('.'):
            # Python split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
            if x.isnumeric():
                # isnumeric 检测字符串是否只由数字组成,这种方法是只针对unicode对象.
                target = target.__getitem__(int(x))
            else:
                target = getattr(target, x)  # getattr() 函数用于返回一个对象属性值。 target = x ?

        # 设置对应的权值
        n_weight = np.prod(target.weight.shape)
        target.weight = value[:n_weight].view(target.weight.shape)
        target.bias = value[n_weight:].view(target.bias.shape)

    def set_weights(self, weights, i=0):
        """输入权值字典，对该网络所有的 MyConv2D 层设置权值"""
        for name, param in weights.items():
            self.set_my_attr(name, weights[name][i])

class RDB(nn.Module):
        def __init__(self, in_channels, num_dense_layer, growth_rate, net):
            """
            :param in_channels: input channel size
            :param num_dense_layer: the number of RDB layers
            :param growth_rate: growth_rate
            """
            super(RDB, self).__init__()
            _in_channels = in_channels
            modules = []
            for i in range(num_dense_layer):
                modules.append(MakeDense(_in_channels, growth_rate))
                _in_channels += growth_rate
            self.residual_dense_layers = nn.Sequential(*modules)
            self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

        def forward(self, x):
            out = self.residual_dense_layers(x)
            out = self.conv_1x1(out)
            out = out + x
            return out
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class MetaNet(nn.Module):
    def __init__(self, param_dict):
        super(MetaNet, self).__init__()  # 对继承自父类的属性进行初始化
        self.param_num = len(param_dict)
        #self.hidden = nn.Linear(320, 128 * self.param_num)
        self.hidden = nn.Linear(320, 128 * self.param_num)
        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
            self.fc_dict[name] = i
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(128, params))

    def forward(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i + 1))
            filters[name] = fc(hidden[:, i * 128:(i + 1) * 128])
        return filters


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return sigmoid(self.net(x).view(batch_size))














