import sys
import os
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import time
import pickle
import math

import Models
from DataGenerator_pytorch import VimeoDataset
from utils import get_psnr, get_optimizer, get_loss, get_name, get_name2, save_batchimages, save_batchflows, save_batchflows_np, save_attention_np, mkdir
import matplotlib.pyplot as plt

import scipy.io

def my_train_net(net, dataloader_train, dataloader_test, config):
    model_name = get_name2(config.model_name, config)
    snapshot_path = config.snapshot_root + model_name + '/'
    mkdir(snapshot_path)
    visualization_path = snapshot_path + config.visualization_folder_train + '/'
    mkdir(visualization_path)

    optimizer = get_optimizer(net, config.optimizer, config.lr, config.weight_decay)
    criterion = get_loss(config.loss)
    psnr_fun = get_loss('PSNRLoss')
    
    # if not os.path.exists(snapshot_path + 'log_train.log'):
        # raise(snapshot_path + 'log_train.log   already exists, consider using a different folder name')
        
    f_train = open(snapshot_path + 'log_train.log', mode='w')
    f_test = open(snapshot_path + 'log_test.log', mode='w')

    if config.train_snapshot_file!='':
        net.load_state_dict(torch.load(snapshot_path+config.train_snapshot_file))

    iter_ = config.checkpoint_epoch * len(dataloader_train)
    try:
        lr = config.lr
        time_start = time.time()
        visualization_path = snapshot_path + config.visualization_folder_train + '/'
        for epoch in range(config.checkpoint_epoch, config.max_epoch+1):
            # train one batch
            for batch_index, batched_sample in enumerate(dataloader_train):
                img0, img1, img2, img1_gt = batched_sample['0'], batched_sample['1'], batched_sample['2'], batched_sample['gt']
                if config.use_gpu:
                    img0, img1, img2, img1_gt = img0.cuda(), img1.cuda(), img2.cuda(), img1_gt.cuda()

                img1_pred, flow_1_1, flow_1_2, flow_1_3, flow_1_4, attention1_1, attention1_2, attention1_3, attention1_4 = net(img0, img1, img2)

                loss = criterion(img1_pred, img1_gt)
                psnr = psnr_fun(img1_pred, img1_gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                time_train = time.time()

                ## display
                if iter_ % config.display==0:
                    print('epoch {}, [{}\{}], lr {:.3e}, psnr {:.4f}, loss {:.4f}, training time {:.3f}'.format(
                            epoch, batch_index, len(dataloader_train), lr, psnr.item(), loss.item(), time.time()-time_start))
                    time_start = time.time()

                # save training result
                if iter_ % 1000 ==0:
                    save_batchimages(img1_pred.data.cpu().numpy(), visualization_path+'predction_', count=iter_*config.batch_size_train)
                    save_batchimages(img1_gt.data.cpu().numpy(), visualization_path+'gt_', count=iter_*config.batch_size_train)
                    save_batchimages(img0.data.cpu().numpy(), visualization_path+'ref_', count=iter_*config.batch_size_train)
                    save_batchimages(img1.data.cpu().numpy(), visualization_path+'img_', count=iter_*config.batch_size_train)
                    save_batchflows(flow_1_1.data.cpu().numpy(), visualization_path+'flow_1_1_', count=iter_*config.batch_size_train)
                    save_batchflows(flow_1_2.data.cpu().numpy(), visualization_path+'flow_1_2_', count=iter_*config.batch_size_train)
                    save_batchflows(flow_1_3.data.cpu().numpy(), visualization_path+'flow_1_3_', count=iter_*config.batch_size_train)
                    save_batchflows(flow_1_4.data.cpu().numpy(), visualization_path+'flow_1_4_', count=iter_*config.batch_size_train)
                    save_batchflows_np(flow_1_1.data.cpu().numpy(), visualization_path+'flow_1_1_', count=iter_*config.batch_size_train)
                    save_batchflows_np(flow_1_2.data.cpu().numpy(), visualization_path+'flow_1_2_', count=iter_*config.batch_size_train)
                    save_batchflows_np(flow_1_3.data.cpu().numpy(), visualization_path+'flow_1_3_', count=iter_*config.batch_size_train)
                    save_batchflows_np(flow_1_4.data.cpu().numpy(), visualization_path+'flow_1_4_', count=iter_*config.batch_size_train)
                    save_attention_np(attention1_1.data.cpu().numpy(), visualization_path+'attention_1_1_', count=batch_index*config.batch_size_test)
                    save_attention_np(attention1_2.data.cpu().numpy(), visualization_path+'attention_1_2_', count=batch_index*config.batch_size_test)
                    save_attention_np(attention1_3.data.cpu().numpy(), visualization_path+'attention_1_3_', count=batch_index*config.batch_size_test)
                    save_attention_np(attention1_4.data.cpu().numpy(), visualization_path+'attention_1_4_', count=batch_index*config.batch_size_test)
                    
                ## logging after each iteration
                f_train.write('{},{},{},{},{}\n'.format(epoch, iter_, lr, psnr.item(), loss.item()))
                f_train.flush()

                iter_ += 1

            ## snapshot
            if (epoch + 1) % config.snapshot_epoch == 0:
                snapshot_filename = snapshot_path + '{}.pth'.format(epoch + 1)
                torch.save(net.state_dict(), snapshot_filename)
                print('Checkpoint {} at epoch {} saved!'.format(snapshot_filename, epoch + 1))
                    
            ## decay lr
            if (epoch + 1) % config.lr_decay_epoch == 0:
                lr = lr * config.lr_decay
                optimizer = get_optimizer(net, config.optimizer, lr, config.weight_decay)
                print('lr decayed to {:.3e} at epoch{}!'.format(lr, epoch + 1))

            ## testing
            if (epoch + 1) % config.testing_epoch == 0:
                with torch.no_grad():
                    visualization_path = snapshot_path + 'epoch_' + str(epoch) + '/'
                    mkdir(visualization_path)

                    print('testing at epoch {}.'.format(epoch + 1))
                    test_psnr_list = list()
                    for batch_index, batched_sample in enumerate(dataloader_test):
                        ## test
                        img0, img1, img2, img1_gt = batched_sample['0'], batched_sample['1'], batched_sample['2'], batched_sample['gt']
                        if config.use_gpu:
                            img0, img1, img2, img1_gt = img0.cuda(), img1.cuda(), img2.cuda(), img1_gt.cuda()

                        img1_pred, flow_1_1, flow_1_2, flow_1_3, flow_1_4, attention1 = net(img0, img1, img2)

                        loss = criterion(img1_pred, img1_gt)
                        psnr = psnr_fun(img1_pred, img1_gt)
                        test_psnr_list.append(psnr.item())

                        # save img1_pred and img1_gt
                        if batch_index*config.batch_size_test<20:
                            save_batchimages(img1_pred.data.cpu().numpy(), visualization_path+'predction_', count=batch_index*config.batch_size_test)
                            save_batchimages(img1_gt.data.cpu().numpy(), visualization_path+'gt_', count=batch_index*config.batch_size_test)
                            save_batchimages(img0.data.cpu().numpy(), visualization_path+'ref_', count=iter_*config.batch_size_train)
                            save_batchflows(flow_1_1.data.cpu().numpy(), visualization_path+'flow_1_1_', count=batch_index*config.batch_size_test)
                            save_batchflows(flow_1_2.data.cpu().numpy(), visualization_path+'flow_1_2_', count=batch_index*config.batch_size_test)
                            save_batchflows(flow_1_3.data.cpu().numpy(), visualization_path+'flow_1_3_', count=batch_index*config.batch_size_test)
                            save_batchflows(flow_1_4.data.cpu().numpy(), visualization_path+'flow_1_4_', count=batch_index*config.batch_size_test)
                            save_batchflows_np(flow_1_1.data.cpu().numpy(), visualization_path+'flow_1_1_', count=batch_index*config.batch_size_test)
                            save_batchflows_np(flow_1_2.data.cpu().numpy(), visualization_path+'flow_1_2_', count=batch_index*config.batch_size_test)
                            save_batchflows_np(flow_1_3.data.cpu().numpy(), visualization_path+'flow_1_3_', count=batch_index*config.batch_size_test)
                            save_batchflows_np(flow_1_4.data.cpu().numpy(), visualization_path+'flow_1_4_', count=batch_index*config.batch_size_test)
                            save_attention_np(attention1_1.data.cpu().numpy(), visualization_path+'attention_1_1_', count=batch_index*config.batch_size_test)
                            save_attention_np(attention1_2.data.cpu().numpy(), visualization_path+'attention_1_2_', count=batch_index*config.batch_size_test)
                            save_attention_np(attention1_3.data.cpu().numpy(), visualization_path+'attention_1_3_', count=batch_index*config.batch_size_test)
                            save_attention_np(attention1_4.data.cpu().numpy(), visualization_path+'attention_1_4_', count=batch_index*config.batch_size_test)

                        ## display
                        if batch_index % 10==0:
                            print('testing at epoch {}, [{}\{}], psnr {:.4f}'.format(epoch + 1, batch_index, len(dataloader_test), psnr.item()))
                    
                    avg_psnr = sum(test_psnr_list) / float(len(test_psnr_list))
                    print('testing at epoch {}, average psnr {:.4f}'.format(epoch + 1 , avg_psnr))
                    f_test.write('{},{}\n'.format(epoch, avg_psnr))
                    f_test.flush()
                    
        f_train.close()
        f_test.close()
    except KeyboardInterrupt:
        torch.save(net.state_dict(), snapshot_path + 'INTERRUPTED.pth')
        print('Saved interrupt')
        f_train.close()
        f_test.close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

def my_test_net(net, dataloader_test, config):
    model_name = get_name2(config.model_name, config)
    snapshot_path = config.snapshot_root + model_name + '/'
    snapshot_file = snapshot_path + config.test_snapshot_file
    visualization_path = snapshot_path + config.visualization_folder_test + '/'

    mkdir(visualization_path)
    print('saving to {}'.format(visualization_path))

    optimizer = get_optimizer(net, config.optimizer, config.lr, config.weight_decay)
    criterion = get_loss(config.loss)
    psnr_fun = get_loss('PSNRLoss')

    # loading
    net.load_state_dict(torch.load(snapshot_path+config.test_snapshot_file))

    ## print model size
    # model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
    # exit()

    ## testing
    with torch.no_grad():
        test_psnr_list = list()
        for batch_index, batched_sample in enumerate(dataloader_test):
            ## test
            img0, img1, img2, img1_gt = batched_sample['0'], batched_sample['1'], batched_sample['2'], batched_sample['gt']
            if config.use_gpu:
                img0, img1, img2, img1_gt = img0.cuda(), img1.cuda(), img2.cuda(), img1_gt.cuda()

            img1_pred, flow_1_1, flow_1_2, flow_1_3, flow_1_4, attention1_1, attention1_2, attention1_3, attention1_4 = net(img0, img1, img2)
            loss = criterion(img1_pred, img1_gt)
            psnr = psnr_fun(img1_pred, img1_gt)
            test_psnr_list.append(psnr.item())

            # save img1_pred and img1_gt
            # if batch_index*config.batch_size_test<100:
            if True:
                save_batchimages(img1_pred.data.cpu().numpy(), visualization_path+'predction_', count=batch_index*config.batch_size_test)
                save_batchimages(img1_gt.data.cpu().numpy(), visualization_path+'gt_', count=batch_index*config.batch_size_test)
                save_batchimages(img0.data.cpu().numpy(), visualization_path+'ref_', count=batch_index*config.batch_size_test)

                save_batchflows(flow_1_1.data.cpu().numpy(), visualization_path+'flow_1_1_', count=batch_index*config.batch_size_test)
                save_batchflows(flow_1_2.data.cpu().numpy(), visualization_path+'flow_1_2_', count=batch_index*config.batch_size_test)
                save_batchflows(flow_1_3.data.cpu().numpy(), visualization_path+'flow_1_3_', count=batch_index*config.batch_size_test)
                save_batchflows(flow_1_4.data.cpu().numpy(), visualization_path+'flow_1_4_', count=batch_index*config.batch_size_test)

                scipy.io.savemat(visualization_path+'data_'+str(batch_index*config.batch_size_test), 
                    {
                        'flow_1_1': flow_1_1.data.cpu().numpy(),
                        'flow_1_2': flow_1_2.data.cpu().numpy(),
                        'flow_1_3': flow_1_3.data.cpu().numpy(),
                        'flow_1_4': flow_1_4.data.cpu().numpy(),
                        'attention_1_1':attention1_1.data.cpu().numpy(),
                        'attention_1_2':attention1_2.data.cpu().numpy(),
                        'attention_1_3':attention1_3.data.cpu().numpy(),
                        'attention_1_4':attention1_4.data.cpu().numpy()
                    },
                    do_compression=True
                )
            ## display
            if batch_index % 10==0:
                print('testing at [{}\{}], psnr {:.4f}'.format(batch_index, len(dataloader_test), psnr.item()))

            if batch_index == 1000:
                break

def get_model_flags_v1():
    parser = argparse.ArgumentParser(description='model descriptions')
    # parser.add_argument('--debug_dataset', type = bool, default=False, help='dubug_dataset')
    
    ''' config '''
    parser.add_argument('--model_name', type = str, default='multi_flow_simple', help='model_name')
    parser.add_argument('--snapshot_root', type = str, default='./snapshots/', help='snapshot_root')
    parser.add_argument('--checkpoint_epoch', type = int, default=0, help='the initial step (for loading model)')
    parser.add_argument('--max_epoch', type = int, default=10, help='max_iter')

    parser.add_argument('--loss', type = str, default='CharbonnierLoss', help= 'loss type')
    parser.add_argument('--lr', type = float, default=0.00003, help='The learning rate: 0.005')
    parser.add_argument('--weight_decay', type = float, default=0.0005, help='The learning rate: 0.005')
    parser.add_argument('--optimizer', type = str, default='Adam', help= 'optimizer')
    parser.add_argument('--batch_size_train', type = int, default=2, help='The batch size for training')
    parser.add_argument('--batch_size_test', type = int, default=2, help='The batch size for training')
    
    parser.add_argument('--lr_decay', type = float, default=0.1, help='lr_decay')
    parser.add_argument('--lr_decay_epoch', type = int, default=4, help='lr_decay_step')
    
    parser.add_argument('--display', type = int, default=10, help='display every n iterations')
    parser.add_argument('--testing_epoch', type = int, default=1, help='snapshot')    
    parser.add_argument('--snapshot_epoch', type = int, default=1, help='snapshot')

    parser.add_argument('--use_gpu', type = bool, default=True, help='use_gpu')
    parser.add_argument('--cuda_devices', type = str, default='0,1,2,3', help='cuda_devices')

    parser.add_argument('--dataset_root', type = str, default='./dataset/vimeo_septuplet/', help= 'dataset_root')

    # preprocessing
    parser.add_argument('--dataset_workers', type = int, default=4, help= 'number of dataset workers')
    parser.add_argument('--dataset_use_crop', type = bool, default=False, help= 'dataset_crop')
    parser.add_argument('--dataset_crop_H', type = int, default=128, help= 'dataset_crop_H')
    parser.add_argument('--dataset_crop_W', type = int, default=128, help= 'dataset_crop_W')

    # test or train
    parser.add_argument('--test_or_train', type = str, default='test', help= 'test/train.')
    parser.add_argument('--train_snapshot_file', type = str, default='', help= 'snapshot filename')
    parser.add_argument('--test_snapshot_file', type = str, default='INTERRUPTED.pth', help= 'snapshot filename')
    parser.add_argument('--visualization_folder_test', type = str, default='visualization test', help= 'visualization_folder folder')
    parser.add_argument('--visualization_folder_train', type = str, default='visualization train', help= 'visualization_folder folder')
    
    return parser.parse_args()

if __name__ == '__main__':
    config = get_model_flags_v1()
    net = Models.FlowNet_MultiscaleWarp_2Frame_multi_Simple(4)
    
    if config.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_devices
        net.cuda()
        #net = nn.DataParallel(net)
        # cudnn.benchmark = True # faster convolutions, but more memory

    ## get dataset
    # dataset preprocessing & augmentation
    augmentation_list = []
    preprocessing_list = []
    if config.dataset_use_crop:
        preprocessing_list += [transforms.RandomCrop((config.dataset_crop_H, config.dataset_crop_W))]
    preprocessing_list +=[transforms.ToTensor()]

    dataset_name = 'denoise'
    if dataset_name=='denoise':
        data_path_corrupted = config.dataset_root +  'sequences_noise/'
        data_path_clean = config.dataset_root + 'sequences/'
        data_list_file_train = config.dataset_root + 'sep_trainlist.txt'
        data_list_file_test = config.dataset_root + 'sep_testlist.txt'
        # data_list_file_train = config.dataset_root + 'sep_trainlist_small.txt'
        # data_list_file_test = config.dataset_root + 'sep_testlist_small.txt'

        composed_train = transforms.Compose(augmentation_list + preprocessing_list)
        composed_test = transforms.Compose(preprocessing_list)

        dataset_train = VimeoDataset(data_path_corrupted, data_path_clean, data_list_file_train, frame_window_size=3, transform=composed_train, small_factor=10000)
        dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size_train, shuffle=True, num_workers=config.dataset_workers)
        dataset_test = VimeoDataset(data_path_corrupted, data_path_clean, data_list_file_test, frame_window_size=3, transform=composed_test, small_factor=1000)
        dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size_test, shuffle=False, num_workers=config.dataset_workers)
    else:
        raise('unknow dataset')

    if config.test_or_train=='test':
        my_test_net(net, dataloader_test, config)
    elif config.test_or_train=='train':
        my_train_net(net, dataloader_train, dataloader_test, config)
    else:
        raise('wrong config.test_or_train args')