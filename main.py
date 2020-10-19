#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
#1215import pandas as pd

from cnn_train_m0314 import CNN_train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Operation-wise Attention Network')
    parser.add_argument('--gpu_num', '-g', type=int, default=0, help='Num. of GPUs')
    parser.add_argument('--mode', '-m', default='yourdata', help='Mode (mix / yourdata)')
    args = parser.parse_args()

    # --- Optimization of the CNN architecture ---
    if args.mode == 'mix':
        cnn = CNN_train(args.mode, imgSize=63, batchsize=32)
        acc = cnn(None, 0, epoch_num=150, gpu_num=args.gpu_num)
    elif args.mode == 'yourdata':
        cnn = CNN_train(args.mode, imgSize=256, batchsize=1)#256 2
        acc = cnn(None, 0, epoch_num=150, gpu_num=args.gpu_num)
