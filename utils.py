import os
import numpy as np
import Loss
import torch.nn as nn
from torch import optim

import math
import imageio
import cv2
from scipy.io import savemat

def get_psnr(img1, img2):
    # print(type(img1), type(img2))
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def get_name(config):
    # lr, weight_decay, optimizer, lr_decay, lr_decay_step, max_iter):
    return 'lr_{}-weight_decay_{}-{}-lr_decay-{}-lr_decay_step-{}_max_iter_{}-batchsize_{}'.format(
            config.lr, config.weight_decay, config.optimizer.upper(), config.lr_decay, config.lr_decay_step, config.max_iter, config.batch_size_train
    )

def get_name2(name, config):
    # lr, weight_decay, optimizer, lr_decay, lr_decay_epoch, max_epoch):
    return name + '_lr_{}-weight_decay_{}-{}-lr_decay-{}-lr_decay_epoch-{}_max_epoch_{}-batchsize_{}'.format(
            config.lr, config.weight_decay, config.optimizer.upper(), config.lr_decay, config.lr_decay_epoch, config.max_epoch, config.batch_size_train
    )

def get_optimizer(net, optimizer, lr, weight_decay):
    if optimizer== 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('optimizer "{}" type not supported'.format(optimizer))
    return optimizer

def get_loss(loss):
    if loss == 'L2Loss':
        criterion = Loss.EuclideanLoss()
    elif loss == 'CharbonnierLoss':
        criterion = Loss.CharbonnierLoss()
    elif loss == 'MSELoss':
        criterion = nn.MSELoss()
    elif loss == 'PSNRLoss':
        criterion = Loss.PsnrLoss()
    else:
        raise ValueError('loss "{}" type not supported'.format(loss))
    return criterion

def save_batchimages(images, path, count=0):
    ''' save batched images to path + str(count + batch_number) + '.png'
    Arg:
        images: numpy object [N, 3, H, W] dtype=float32 range(0-1)
        path: the path
    '''
    N = images.shape[0]
    images = (images.clip(0.0, 1.0)*255.0).astype(np.uint8).transpose(0,2,3,1)
    
    for i in range(N):
        img = images[i,:,:,:]
        imageio.imwrite(path+str(count + i) + '.png', img)

def vis_flow(flow):
    ''' 
    Arg: flow: [H, W, 2]
    ''' 
    # print(flow.shape)
    H, W, _ = flow.shape

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros((H,W,3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # print(mag.shape, ang.shape, hsv.shape)
    # exit()
    
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def save_batchflows(flows, path, count=0):
    ''' save visualized flows to path + str(count + batch_number) + '.png'
    Arg:
        flows: numpy object [N, 2, H, W] dtype=float32 range(-inf, inf)
        path: the path
    '''
    N = flows.shape[0]
    flows = flows.transpose(0,2,3,1)
    
    for i in range(N):
        flow = flows[i,:,:,:]
        imageio.imwrite(path+str(count + i) + '.png', vis_flow(flow))


def save_batchflows_np(flows, path, count=0):
    ''' save visualized flows to path + str(count + batch_number) + '.png'
    Arg:
        flows: numpy object [N, 2, H, W] dtype=float32 range(-inf, inf)
        path: the path
    '''
    N = flows.shape[0]
    flows = flows.transpose(0,2,3,1)
    
    for i in range(N):
        flow = flows[i,:,:,:]
        savemat(path+str(count + i) + '.mat', {'flow':flow})


def save_attention_np(flows, path, count=0):
    ''' save visualized flows to path + str(count + batch_number) + '.png'
    Arg:
        flows: numpy object [N, 2, H, W] dtype=float32 range(-inf, inf)
        path: the path
    '''
    N = flows.shape[0]
    
    for i in range(N):
        flow = flows[i,:,:,:]
        savemat(path+str(count + i) + '.mat', {'flow':flow})

def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)