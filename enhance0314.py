
import torch
from torch import nn
import torch.nn.functional as F


class Dehaze(nn.Module):
    def __init__(self,channles):
        super(Dehaze, self).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.tanh=nn.Tanh()

        self.refine1= nn.Conv2d(channles, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x,weights):
        dehaze = self.relu((self.refine1(x))) * weights[:, :, 24].view([-1, 1, 1, 1])
        #dehaze1 = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out) * weights[:, :, 25].view([-1, 1, 1, 1])
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out) * weights[:, :, 26].view([-1, 1, 1, 1])
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out) * weights[:, :, 27].view([-1, 1, 1, 1])
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out) * weights[:, :, 28].view([-1, 1, 1, 1])

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))

        return dehaze

class Dehaze1(nn.Module):
    def __init__(self,channles):
        super(Dehaze1, self).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.tanh=nn.Tanh()

        self.refine1= nn.Conv2d(channles, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x,weights):
        dehaze = self.relu((self.refine1(x)))* weights[:, :, 29].view([-1, 1, 1, 1])
       # dehaze1 = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 16)

        x102 = F.avg_pool2d(dehaze, 8)

        x103 = F.avg_pool2d(dehaze, 4)

        x104 = F.avg_pool2d(dehaze, 2)
        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out) * weights[:, :, 30].view([-1, 1, 1, 1])
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out) * weights[:, :, 31].view([-1, 1, 1, 1])
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out) * weights[:, :, 32].view([-1, 1, 1, 1])
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out) * weights[:, :, 33].view([-1, 1, 1, 1])

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))

        return dehaze