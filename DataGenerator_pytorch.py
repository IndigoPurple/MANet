import numpy as np
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

class VimeoDataset(Dataset):
    ''' Vimeo Dataset '''

    def __init__(self, data_path_corrupted, data_path_clean, data_list_file, frame_window_size=3, transform=None, small_factor=0):
        assert frame_window_size%2==1, "frame_window_size should be odd"
        ## file list
        self.data_list_file = data_list_file
        self.folder_list_corrupted = self.get_list(data_path_corrupted)
        self.folder_list_clean = self.get_list(data_path_clean)
        self.file_list = ['im'+str(i)+'.png' for i in range(1,8)]
        # transform
        self.transform = transform
        self.crop = None
        # 
        if small_factor!=0:
            self.M = small_factor
        else:
            self.M = len(self.folder_list_clean)
        self.max_frame = len(self.file_list) # frame number
        self.K = self.max_frame - frame_window_size + 1 # frame start index number
        self.N = self.M * self.K    # total number
        self.frame_window_size = frame_window_size
        self.half_frame_window_size = frame_window_size//2
        # get shape
        _, self.H, self.W = self.__getitem__(0)['gt'].size()

    def get_list(self, data_path):
        folder_list = list()
        with open(self.data_list_file, 'r') as f_index:
            reader = csv.reader(f_index)
            for row in reader:
                if row:
                    folder_list.append(data_path + row[0] + '/')
        return folder_list

    def get_frame(self, m, f, use_clean=True):
        ''' return a 3-D [3,H,W] float32 [0.0-1.0] tensor '''
        if use_clean:
            filename = self.folder_list_clean[m] + self.file_list[f]
            image = Image.open(filename)
            return image
        else:
            # filename = self.folder_list_corrupted[m] + self.file_list[f]
            filename = self.folder_list_clean[m] + self.file_list[f]                # add noise manully
            image = Image.open(filename)
            image_np = np.array(image)      # to HWC image, uint8
            image_noise_np = image_np + np.random.normal(loc=0.0, scale=25.5,size=image_np.shape).astype(np.uint8)
            mask_np = (np.random.uniform(size=(image_np.shape[0],image_np.shape[1],1)) > 0.1).astype(np.uint8)
            mask_np = np.tile(mask_np,(1,1,3))
            image_noise_final_np = mask_np * image_np + (1-mask_np) * image_noise_np
            image_noise = Image.fromarray(image_noise_final_np)
            return image_noise

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # initialize seed for every sample
        if self.transform:
            seed = np.random.randint(2147483647)

        m = idx // self.K
        f_st = idx % self.K
        # build dict
        sample = dict()
        for inc in range(0, self.frame_window_size):
            sample[str(inc)] = self.get_frame(m, f_st+inc, use_clean=False)
            random.seed(seed)
            if self.transform:
                sample[str(inc)]  = self.transform(sample[str(inc)])
        
        sample['gt'] = self.get_frame(m, f_st + self.half_frame_window_size, use_clean=True)
        random.seed(seed)
        if self.transform:
            sample['gt'] = self.transform(sample['gt'])

        return sample

if __name__ == "__main__":
    data_path_corrupted = './dataset/vimeo_septuplet/sequences_noise/'
    data_path_clean = './dataset/vimeo_septuplet/sequences/'
    data_list_file = './dataset/vimeo_septuplet/sep_trainlist.txt'
    composed = transforms.Compose([transforms.RandomCrop((128,128)),
                                     transforms.ToTensor()])
    dataset = VimeoDataset(data_path_corrupted, data_path_clean, data_list_file, frame_window_size=3, transform=composed)
    dataset[0]
    exit()

    #### test pytorch dataset
    # print(len(dataset))
    
    # fig = plt.figure()
    # plt.axis('off')
    # plt.ioff()
    # im = plt.imshow(np.zeros((dataset.H, dataset.W, 3)), vmin=0, vmax=1)

    # for i in range(len(dataset)-1, 0, -1):
    #     sample = dataset[i]
    #     for t in sample:
    #         print(t, sample[t].size())
    #         im.set_data(sample[t].numpy().transpose(1,2,0))
    #         plt.pause(0.1)
    # exit()

    #### test dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=6)
    print(len(dataset), len(dataloader))

    for i_batch, sample_batched in enumerate(dataloader):
        # print(i_batch, sample_batched['gt'].size())
        # visualization
        images_batch = sample_batched['gt']
        batch_size = images_batch.size()[0]
        im_size = images_batch.size()[1:]
        print(batch_size, im_size)

        grid = utils.make_grid(images_batch, nrow=2)
        plt.imshow(grid.numpy().transpose(1,2,0))
        plt.show()

        # observe 4th batch and stop.
        # if i_batch == 3:
        #     plt.figure()
        #     show_landmarks_batch(sample_batched)
        #     plt.axis('off')
        #     plt.ioff()
        #     plt.show()
        #     break