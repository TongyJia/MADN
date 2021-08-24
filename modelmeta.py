#from wban import wbal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



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

class Meta(nn.Module):

      def __init__(self,base):  # 初始化,传入相应的参数
        super(Meta, self).__init__()
        self.base = base
        self.conv1 = nn.Conv2d(12, base*2, kernel_size=3, stride = 1,padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(base*2, base*4, kernel_size=3,stride = 1, padding=1)
        self.bn2 = nn.BatchNorm2d(base*4)
        self.relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(base*4, base*8, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(base * 8)
        self.relu3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(base*8, base*4, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(base * 4)
        self.relu4 = nn.LeakyReLU(0.2)
        self.conv5 = nn.Conv2d(base*4, base*2, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(base * 2)
        self.relu5 = nn.LeakyReLU(0.2)
        self.conv6 = nn.Conv2d(base * 2, 3, kernel_size=3, stride=1, padding=1)
        #self.bn6 = nn.BatchNorm2d(3)
        #self.relu6 = nn.LeakyReLU(0.2)



      def forward(self, x):
          #conv1 = self.conv1(x)
          fea1 = self.relu1(self.conv1(x))
          #conv2 = self.conv2(relu1)
          #bn2 = self.bn2(self.conv2(relu1))
          fea2 = self.relu2(self.bn2(self.conv2(fea1)))

          #conv3 = self.conv2(relu2)
          #bn3 = self.bn2(self.conv2(relu2))
          fea3 = self.relu3(self.bn3(self.conv3(fea2)))
          #conv4 = self.conv2(relu3)
          #bn4 = self.bn4(self.conv2(relu3))
          fea4 = self.relu4(self.bn4(self.conv4(fea3)))
          #conv5 = self.conv2(relu4)
          #bn5 = self.bn5(self.conv2(relu4))
          fea5 = self.relu5(self.bn5(self.conv5(fea4)))
          fea6 = self.conv6(fea5)
          '''
          print(fea1.size())
          print(fea2.size())
          print(fea3.size())
          print(fea4.size())
          print(fea5.size())
          print(fea6.size())
          '''


          return fea1,fea2,fea3,fea4,fea5,fea6
'''
#model=fea(16)
model = models.vgg16(pretrained=True)
# Initialize optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)

print("Model's state_dict:")
# Print model's state_dict
for param_tensor in model.state_dict():
    print(param_tensor,"\t",model.state_dict()[param_tensor].size())
print("optimizer's state_dict:")
# Print optimizer's state_dict
for var_name in optimizer.state_dict():
    print(var_name,"\t",optimizer.state_dict()[var_name])





import torch
import torch.nn as nn
import torch.nn.functional as F
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def farward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
# Initialize model
model=TheModelClass()
# Initialize optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)

print("Model's state_dict:")
# Print model's state_dict
for param_tensor in model.state_dict():
    print(param_tensor,"\t",model.state_dict()[param_tensor].size())
print("optimizer's state_dict:")
# Print optimizer's state_dict
for var_name in optimizer.state_dict():
    print(var_name,"\t",optimizer.state_dict()[var_name])
'''
















