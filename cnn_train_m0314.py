#!/usr/bin/env python
# -*- coding: utf-8 -*-


#2:21:47

# l1_loss = L1_loss(output, hr_patch)
# perceptual_loss = 0.04 * loss_network(output, hr_patch)
# loss = l1_loss+perceptual_loss
import time
import math
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from skimage.measure import compare_ssim as ski_ssim
import os
import csv
import logging

import utils
from data_utils0210 import *
from tqdm import tqdm
from pregrocess import *
from model200314 import *
from torchvision.models import vgg16
from perceptual import LossNetwork


#meta = torch.load("./model/dehaze_80.pth",map_location="cuda:0")
meta = torch.load("./meta/dehaze_80.pth",map_location="cuda:0")
prepro=pre(0.5,0)

class CNN_train():
    def __init__(self, dataset_name, imgSize=128, batchsize=8):
        self.imgSize = imgSize
        self.batchsize = batchsize
        self.dataset_name = dataset_name

        # load dataset
        if dataset_name == 'mix' or dataset_name == 'yourdata':
            if dataset_name == 'mix':
                self.num_work = 8
                train_dir = '../0907meta-dehaze/aodnet/hazedataset/haze/'
                val_dir = '../0907meta-dehaze/aodnet/hazedataset/haze1/'
                test_dir = '../0907meta-dehaze/aodnet/hazedataset/haze2/'
                #train_set = get_dataset_deform(train_dir, val_dir, test_dir, 0)
               # val_set = get_dataset_deform(train_dir, val_dir, test_dir, 1)
                # test_set = get_dataset_deform(train_dir, val_dir, test_dir, 2)
                #self.dataloader = DataLoader(dataset=train_set, num_workers=self.num_work, batch_size=self.batchsize, shuffle=True, pin_memory=True)
                #self.val_loader = DataLoader(dataset=val_set, num_workers=self.num_work, batch_size=1, shuffle=False, pin_memory=False)
                # self.test_dataloader = DataLoader(dataset=test_set, num_workers=self.num_work, batch_size=1, shuffle=False, pin_memory=False)
            elif dataset_name == 'yourdata':
                self.num_work = 8

                train_set = TrainDatasetFromFolder1('/home/omnisky/volume/ITSV2/hazy','/home/omnisky/volume/ITSV2/clear10', crop_size=self.imgSize )

                test_set = TestDatasetFromFolder1(
                    '/home/omnisky/volume/0dehaze-project/GridDehazeNet-master/data/test/SOTS/indoor')
                #train_set = get_training_set(train_input_dir, train_target_dir, True)
                #test_set = get_training_set(test_input_dir, test_target_dir, False)


                self.dataloader = DataLoader(dataset=train_set, num_workers=self.num_work, batch_size=self.batchsize, shuffle=True, drop_last=True)
                #self.test_dataloader = DataLoader(dataset=test_set, num_workers=self.num_work, batch_size=1, shuffle=False)
                self.val_loader = DataLoader(dataset=test_set, num_workers=self.num_work, batch_size=1,
                                                  shuffle=False)
        else:
            print('\tInvalid input dataset name at CNN_train()')
            exit(1)

    def __call__(self, cgp, gpuID, epoch_num=150, gpu_num=1):
        print('GPUID    :', gpuID)
        print('epoch_num:', epoch_num)
        
        # define model
        torch.manual_seed(2018)
        torch.cuda.manual_seed(2018)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        L1_loss = nn.L1Loss()
        L1_loss = L1_loss.cuda(gpuID)
        # --- Define the perceptual loss network --- #
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to(device)
        for param in vgg_model.parameters():
            param.requires_grad = False

        loss_network = LossNetwork(vgg_model).cuda(gpuID)
        loss_network.eval()

        model = TransformNet(32).to(device)

        if gpu_num > 1:
            device_ids = [i for i in range(gpu_num)]
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda(gpuID)
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        print('Param:', utils.count_parameters_in_MB(model))
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
        test_interval =1 #5
        # for output images
        if not os.path.exists('./results'):
            os.makedirs('./results/Inputs')
            os.makedirs('./results/Outputs')
            os.makedirs('./results/Targets')

        # Train loop

        for epoch in range(1, epoch_num+1):
            scheduler.step()
            start_time = time.time()
            print('epoch', epoch)
            train_loss = 0
            train_bar = tqdm(self.dataloader)
            running_results = {'batch_sizes': 0, 'loss': 0}
            ite = 0
            for module in model.children():
                module.train(True)
            for input, target in train_bar:#enumerate(self.dataloader):
              if target.size(1)==3:
                ite+=1
                lr_patch = Variable(input, requires_grad=False).cuda(gpuID)
                hr_patch = Variable(target, requires_grad=False).cuda(gpuID)
                batch_size = input.size(0)
                running_results['batch_sizes'] += batch_size

                unloader = transforms.ToPILImage()
                image = input.cpu().clone()
                image = image.squeeze(0)
                haze = unloader(image)
                x = cv.cvtColor(np.asarray(haze), cv.COLOR_RGB2BGR)
                pre_haze = prepro(x).cuda(gpuID)

                outs = []

                fea1, fea2, fea3, fea4, fea5, fea6 = meta(pre_haze)
                outs.append(fea1)
                outs.append(fea2)
                outs.append(fea3)
                outs.append(fea4)
                outs.append(fea5)

                haze_features = cat((fea1, fea2, fea3, fea4, fea5), 1)

                output = model(lr_patch, haze_features, pre_haze)#output = model(pre_haze, haze_features)

                dimage = output.cpu().clone()
                dimage = dimage.squeeze(0)
                dhaze = unloader(dimage)
                dx = cv2.cvtColor(np.asarray(dhaze), cv2.COLOR_RGB2BGR)
                pre_dhaze = prepro(dx).to(device)
                de_f1, de_f2, de_f3, de_f4, de_f5, de_f6 = meta(pre_dhaze)

                simage = target.cpu().clone()
                simage = simage.squeeze(0)
                src = unloader(simage)
                src = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
                pre_src = prepro(src).to(device)
                src_f1, src_f2, src_f3, src_f4, src_f5, src_f6 = meta(pre_src)

                optimizer.zero_grad()
                l1_loss = L1_loss(output, hr_patch)
                perceptual_loss = loss_network(output, hr_patch)
                fe_loss = (
                        F.mse_loss(de_f1, src_f1) + F.mse_loss(de_f2, src_f2) + F.mse_loss(de_f3, src_f3) +
                        F.mse_loss(de_f4, src_f4) + F.mse_loss(de_f5, src_f5) )#+ F.mse_loss(de_f6, src_f6)

                loss = l1_loss + 0.04 * fe_loss +0.04 * perceptual_loss # +perceptual_loss
                #loss = l1_loss+0.04 * perceptual_loss
                #l1_loss.backward()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                running_results['loss'] += loss.item()
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, 150, running_results['loss'] / running_results['batch_sizes']))
                if ite % 100 == 0:
                    #print(output)
                    vutils.save_image(lr_patch.data, './input_sample%d.png' % gpuID, normalize=True)
                    vutils.save_image(hr_patch.data, './target_sample%d.png' % gpuID, normalize=True)#True
                    vutils.save_image(output.data, './output_sample%d.png' % gpuID, normalize=True)
                    #print(output)

            print('Train set : Average loss: {:.4f}'.format(train_loss))
            print('time ', time.time()-start_time)
            torch.save(model.state_dict(), './model_%d.pth' % int(epoch))

            # check val/test performance
            if epoch % test_interval == 0:
                with torch.no_grad():
                    print('------------------------')
                    for module in model.children():
                        module.train(False)
                    test_psnr = 0
                    test_ssim = 0
                    eps = 1e-10
                    test_ite = 0
                    #for image_name,input, target in enumerate(self.val_loader):
                    for i, (input, target) in enumerate(self.val_loader):
                        lr_patch = Variable(input, requires_grad=False).cuda(gpuID)
                        hr_patch = Variable(target, requires_grad=False).cuda(gpuID)

                        unloader = transforms.ToPILImage()
                        image = input.cpu().clone()
                        image = image.squeeze(0)
                        haze = unloader(image)
                        x = cv.cvtColor(np.asarray(haze), cv.COLOR_RGB2BGR)
                        pre_haze = prepro(x).cuda(gpuID)

                        outs = []

                        fea1, fea2, fea3, fea4, fea5, fea6 = meta(pre_haze)
                        outs.append(fea1)
                        outs.append(fea2)
                        outs.append(fea3)
                        outs.append(fea4)
                        outs.append(fea5)

                        haze_features = cat((fea1, fea2, fea3, fea4, fea5), 1)

                        output = model(lr_patch, haze_features,pre_haze)#output = model(lr_patch,haze_features )
                        #print(output)
                        # save images
                        vutils.save_image(output.data, './results/Outputs/%05d.png' % (int(i)), padding=0, normalize=True)#False
                        vutils.save_image(lr_patch.data, './results/Inputs/%05d.png' % (int(i)), padding=0, normalize=True)
                        vutils.save_image(hr_patch.data, './results/Targets/%05d.png' % (int(i)), padding=0, normalize=True)
                        # Calculation of SSIM and PSNR values
                        #print(output)
                        output = output.data.cpu().numpy()[0]
                        output[output>1] = 1
                        output[output<0] = 0
                        output = output.transpose((1,2,0))
                        hr_patch = hr_patch.data.cpu().numpy()[0]
                        hr_patch[hr_patch>1] = 1
                        hr_patch[hr_patch<0] = 0
                        hr_patch = hr_patch.transpose((1,2,0))
                        # SSIM
                        test_ssim+= ski_ssim(output, hr_patch, data_range=1, multichannel=True)
                        # PSNR
                        imdf = (output - hr_patch) ** 2
                        mse = np.mean(imdf) + eps
                        test_psnr+= 10 * math.log10(1.0/mse)
                        test_ite += 1
                    test_psnr /= (test_ite)
                    test_ssim /= (test_ite)
                    print('Valid PSNR: {:.4f}'.format(test_psnr))
                    print('Valid SSIM: {:.4f}'.format(test_ssim))
                    f = open('PSNR.txt', 'a')
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([epoch, test_psnr, test_ssim])
                    f.close()
                    print('------------------------')
                torch.save(model.state_dict(), './model_%d.pth' % int(epoch))

        return train_loss
#out1: