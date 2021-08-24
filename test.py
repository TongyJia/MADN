#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from skimage.measure import compare_ssim as ski_ssim
from model200314 import *
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
#from data_load_own import get_training_set, get_test_set
#from data_load_mix import get_dataset_deform
import argparse
from pregrocess import pre
from data_utils0210 import  TestDatasetFromFolder2,TestDatasetFromFolder,TestDatasetFromFolder4
from tqdm import tqdm
import time
from skimage import transform

parser = argparse.ArgumentParser(description='Operation-wise Attention Network')
parser.add_argument('--gpu_num', '-g', type=int, default=0, help='Num. of GPUs')
parser.add_argument('--mode', '-m', default='yourdata', help='Mode (mix / yourdata)')


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load dataset
if args.mode == 'mix' or args.mode == 'yourdata':
    if args.mode == 'mix':
        num_work = 8
        train_dir = '/dataset/train/'
        val_dir = '/dataset/val/'
        test_dir = './dataset/test/'
    elif args.mode == 'yourdata':
        num_work = 1
        test_set = TestDatasetFromFolder4('/home/omnisky/4t/RESIDE/RTTS/RTTS')#
        test_dataloader = DataLoader(dataset=test_set, num_workers=num_work, batch_size=1, shuffle=False, pin_memory=False)
        test_bar = tqdm(test_dataloader, desc='[testing datasets]')
else:
    print('\tInvalid input dataset name at CNN_train()')
    exit(1)

prepro=pre(0.5,0)
meta = torch.load("./dehaze_80.pth",map_location="cuda:0")#,map_location="cuda:0"
model = TransformNet(32)
model.load_state_dict(torch.load('./model.pth'))
model = model.cuda(0)

# model
gpuID = 0
torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

if not os.path.exists('./results_r'):  #save path
    os.makedirs('./results_r/Inputs')
    os.makedirs('./results_r/Outputs')
    os.makedirs('./results_r/Targets')


test_ite = 0
test_psnr = 0
test_ssim = 0
eps = 1e-10
start_time = time.time()
tt = 0

for image_name, input, target in test_bar:
    #print(input.size())
    #print(target.size())
    image_name = image_name[0]
    lr_patch = Variable(input, requires_grad=False).cuda(gpuID)
    hr_patch = Variable(target, requires_grad=False).cuda(gpuID)

    unloader = transforms.ToPILImage()
    image = lr_patch.cpu().clone()
    image = image.squeeze(0)
    haze = unloader(image)
    x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)

    pre_haze = prepro(x).to(device)
    # start_time = time.time()
    t0 = time.time()
    fea1, fea2, fea3, fea4, fea5, fea6 = meta(pre_haze)
    haze_features = cat((fea1, fea2, fea3, fea4, fea5), 1)
    # start_time = time.time()
    output = model(lr_patch, haze_features, pre_haze)
    t1 = time.time()
    tt = tt + t1- t0
    print(str((t1 - t0)))
    # save images
    vutils.save_image(output.data, './results_r/Outputs/' + image_name, padding=0, normalize=True)
    vutils.save_image(lr_patch.data, './results_r/Inputs/' + image_name, padding=0, normalize=True)
    vutils.save_image(hr_patch.data, './results_r/Targets/'+ image_name, padding=0, normalize=True)
    # SSIM and PSNR
    #print(output)
    output = output.data.cpu().numpy()[0]
    output[output>1] = 1
    output[output<0] = 0
    output = output.transpose((1,2,0))
    #print(output)
    hr_patch = hr_patch.data.cpu().numpy()[0]
    hr_patch[hr_patch>1] = 1
    hr_patch[hr_patch<0] = 0
    hr_patch = hr_patch.transpose((1,2,0))
    # SSIM
    ssim = ski_ssim(output, hr_patch, data_range=1, multichannel=True)
    test_ssim+=ssim
    # PSNR
    imdf = (output - hr_patch) ** 2
    mse = np.mean(imdf) + eps
    psnr = 10 * math.log10(1.0/mse)

    test_psnr+= psnr
    test_ite += 1
    print('PSNR: {:.4f}'.format(psnr))
    print('SSIM: {:.4f}'.format(ssim))

print('mt:', test_ite, tt/test_ite)
test_psnr /= (test_ite)
test_ssim /= (test_ite)
print('Test mPSNR: {:.4f}'.format(test_psnr))
print('Test mSSIM: {:.4f}'.format(test_ssim))
print('------------------------')


