import numpy as np
import h5py
import csv
import imageio
import matplotlib.pyplot as plt

def generator_image_list(index_file, dataset_root='./'):
    with open(index_file, 'r') as f_index:
        reader = csv.reader(f_index)
        for row in reader:
            path = dataset_root + '/' + row[0] + '/im'
            image_list = []
            try:
                for i in range(1,8):
                    image_name = path + str(i) + '.png'
                    image = imageio.imread(image_name)
                    image_list.append(image)
                yield(image_list, row[0])
            except FileNotFoundError:
                print('   {}__.png not found...'.format(path))
                yield(list(), row[0])

def generator_image_pair(img_list):
    for i in range(len(img_list)-1):
        im1 = img_list[i]
        im2 = img_list[i+1]
        yield((im1,im2))


def gen_hdf5_pair(h5_filename, dataset_list_file, dataset_root='./', H=256, W=448, C=3, chunk_size=64, key='im'):
    '''
        generate pair image
    '''
    with h5py.File(h5_filename, 'w') as f_h5:
        f_h5.create_dataset('/N', (1,) , dtype = np.int64)
        f_h5.create_dataset(key+'1',(0,H,W,C) , dtype = np.uint8 , maxshape = (None,H,W,C), chunks = (chunk_size,H,W,C))
        f_h5.create_dataset(key+'2',(0,H,W,C) , dtype = np.uint8 , maxshape = (None,H,W,C), chunks = (chunk_size,H,W,C))
        count = 0

        for image_list, path in generator_image_list(dataset_list_file, dataset_root):
            print(path)
            for im1,im2 in generator_image_pair(image_list):
                # before adding, check h5 space
                if count%chunk_size == 0:
                    f_h5[key+'1'].resize(count+chunk_size,axis=0)
                    f_h5[key+'2'].resize(count+chunk_size,axis=0)
                # add data
                f_h5[key+'1'][count,...]=im1
                f_h5[key+'2'][count,...]=im2
                count+=1
                f_h5['/N'][...]=count+1

                ## debug
                # print(im1.shape)
                # plt.imshow(im1)
                # plt.draw()
                # plt.pause(0.01)

def gen_hdf5_5d(h5_filename, dataset_list_file, dataset_root='./', H=256, W=448, C=3, F=7, chunk_size=64, key='im'):
    '''
        generate video array
    '''
    with h5py.File(h5_filename, 'w') as f_h5:
        f_h5.create_dataset('/N', (1,) , dtype = np.int64)
        f_h5.create_dataset(key, (0,F,H,W,C), dtype = np.uint8 , maxshape = (None,F,H,W,C), chunks = (chunk_size,F,H,W,C))

        count = 0

        for image_list, path in generator_image_list(dataset_list_file, dataset_root):
            # before adding, check h5 space
            if count%chunk_size == 0:
                f_h5[key].resize(count+chunk_size,axis=0)
            # fill data
            for frame, im in enumerate(image_list):
                f_h5[key][count,frame,...]=im
                ## debug
                # print(count, frame)
                # print(im1.shape)
                # plt.imshow(im)
                # plt.draw()
                # plt.pause(0.01)

            # increase count
            count+=1
            f_h5['/N'][...]=count+1


if __name__ == "__main__":
    ''' paried data format '''
    # gen_hdf5_pair('../dataset_h5/train_data.h5', '../dataset/vimeo_septuplet/sep_trainlist.txt', dataset_root='../dataset/vimeo_septuplet/sequences', key='/im')
    # gen_hdf5_pair('../dataset_h5/test_data.h5', '../dataset/vimeo_septuplet/sep_testlist.txt', dataset_root='../dataset/vimeo_septuplet/sequences', key='/im')
    
    # gen_hdf5_pair('../dataset_h5/train_data.h5', '../dataset/vimeo_septuplet/sep_trainlist.txt', dataset_root='../dataset/vimeo_septuplet/sequences_noise', key='/im_noise')
    # gen_hdf5_pair('../dataset_h5/test_data.h5', '../dataset/vimeo_septuplet/sep_testlist.txt', dataset_root='../dataset/vimeo_septuplet/sequences_noise', key='/im_noise')

    ''' 5-d array data format '''
    gen_hdf5_5d('../dataset_h5/train_data.h5', '../dataset/vimeo_septuplet/sep_trainlist.txt', dataset_root='../dataset/vimeo_septuplet/sequences', key='/im')
    # gen_hdf5_5d('../dataset_h5/train_data.h5', '../dataset/vimeo_septuplet/sep_trainlist.txt', dataset_root='../dataset/vimeo_septuplet/sequences_noise', key='/im_noise')
    # gen_hdf5_5d('../dataset_h5/train_data.h5', '../dataset/vimeo_septuplet/sep_trainlist.txt', dataset_root='../dataset/vimeo_septuplet/sequences_noise', key='/im_noise')

