import cv2 as cv

from PIL import Image
from PIL import ImageEnhance
from utils import *

from torch import nn, cat


class enhance(nn.Module):

    def __init__(self,a,b):
        super(enhance, self).__init__()
        self.a = a
        self.b = b

    def forward(self, img):
        enh_con = ImageEnhance.Contrast(img)
        contrast = 1.5
        img_ec = enh_con.enhance(contrast)


        #y = np.float(self.a) * img + self.b
        #y[y > 255] = 255
        #y = np.round(y)
        #img_ec = y.astype(np.uint8)

        img_ec = transforms.ToTensor()(img_ec)


        return img_ec


class balance(nn.Module):

    def __init__(self):
        super(balance, self).__init__()


    def forward(self, img):
        r, g, b = cv.split(img)
        r_avg = cv.mean(r)[0]
        g_avg = cv.mean(g)[0]
        b_avg = cv.mean(b)[0]
        # 求各个通道所占增益
        k = (r_avg + g_avg + b_avg) / 3
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg
        r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        balance_img = cv.merge([b, g, r])
        balance_img = transforms.ToTensor()(balance_img)
        return balance_img


class gamma(nn.Module):

    def __init__(self):
        super(gamma, self).__init__()


    def forward(self, img):
        img_norm = img / 255.0  # 注意255.0得采用浮点数
        img_gamma = np.power(img_norm, 2.5) * 255.0
        img_gamma = img_gamma.astype(np.uint8)
        img_gamma = transforms.ToTensor()(img_gamma)

        return img_gamma


class pre(nn.Module):

    def __init__(self,a,b):
        super(pre, self).__init__()
        self.a = a
        self.b = b

    def forward(self, img):


        y = np.float(self.a) * img + self.b
        y[y > 255] = 255
        y = np.round(y)
        img_bright = y.astype(np.uint8)

        img_norm = img / 255.0  # 注意255.0得采用浮点数
        img_gamma = np.power(img_norm, 2.5) * 255.0
        img_gamma = img_gamma.astype(np.uint8)

        r, g, b = cv.split(img)
        r_avg = cv.mean(r)[0]
        g_avg = cv.mean(g)[0]
        b_avg = cv.mean(b)[0]
        # 求各个通道所占增益
        k = (r_avg + g_avg + b_avg) / 3
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg
        r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        balance_img = cv.merge([b, g, r])

        balance_img = transforms.ToTensor()(balance_img)
        img_bright = transforms.ToTensor()(img_bright)
        img_gamma = transforms.ToTensor()(img_gamma)
        img = transforms.ToTensor()(img)

        hazef = cat([img,balance_img, img_bright, img_gamma], 0)

        height = hazef.shape[1]
        width = hazef.shape[2]
        hazef = hazef.expand(1, 12, height, width)


        return  hazef
