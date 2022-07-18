import numpy as np
import csv
import imageio
import matplotlib.pyplot as plt

class DataGenerator:
    '''
        generate batch data for training
    '''
    def __init__(self, data_path_corrupted, data_path_clean, data_list_file, batch_size = 4, frame_window_size=3, shuffle=False, crop=None, seed=10):
        assert frame_window_size%2==1, "frame_window_size should be odd"

        self.data_path_corrupted = data_path_corrupted
        self.data_path_clean = data_path_clean

        self.data_list_file = data_list_file
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        ## file list
        self.folder_list_corrupted = self.get_list(data_path_corrupted)
        self.folder_list_clean = self.get_list(data_path_clean)
        self.file_list = ['im'+str(i)+'.png' for i in range(1,8)]
        
        ## get size infos
        self.N_max = len(self.folder_list_clean)
        self.frame_max = len(self.file_list)
        self.frame_window_size = frame_window_size
        self.half_frame_window_size = frame_window_size//2

        # crop config
        self.crop = crop
        if crop is None:        
            _, self.H, self.W = self.get_frame(0, 0).shape
        else:
            self.H, self.W = crop

        ## iteration mode (shuffle mode) 
        if shuffle:
            self.index_iterable = self.shuffle_iterator()
        else:
            self.index_iterable = self.non_shuffle_iterator()

    def yield_batches(self):
        ''' main method for training!!!!
            yields a list of (frame_window_size + 1) frames of tensors, 
                        corrspoinding to frame_window_size corrpted, and 1 clean
                        shape=[batch_size, 3, H, W], dtype=float32, range=[0.0-1.0]
        '''
        index_iterable = self.index_iterable
        batch_size = self.batch_size

        batched_frames = list()
        for f in range(self.frame_window_size+1):
            batched_frames.append(np.zeros([batch_size, 3, self.H, self.W], dtype=np.float32))
        
        count = 0
        for n, f_start in index_iterable:
            # print('iter->',n,f_start,count)
            frames = self.get_frame_window(n, f_start, self.frame_window_size)
            for f in range(self.frame_window_size):
                batched_frames[f][count, ...] = frames[f][...]
            frame_clean = self.get_frame(n, f_start+self.half_frame_window_size)
            batched_frames[-1][count, ...] = frame_clean

            count = (count+1) % batch_size
            if count == 0:
                yield batched_frames

    def get_frame_window(self, n, f, w):
        ''' return w [3,H,W] float32 [0.0-1.0] tensors '''
        l = list()
        for f in range(f, f+w):
            filename = self.folder_list_corrupted[n] + self.file_list[f]
            # print(filename)
            image = imageio.imread(filename).astype(np.float32)/255.0
            if self.crop is not None:
                image = image[0:self.H, 0:self.W, :]
            l.append(image.transpose([2,0,1]))
        return l

    def get_frame(self, n, f):
        ''' return a 3-D [3,H,W] float32 [0.0-1.0] tensor '''
        filename = self.folder_list_clean[n] + self.file_list[f]
        # print(filename)
        image = imageio.imread(filename).astype(np.float32)/255.0
        if self.crop is not None:
            image = image[0:self.H, 0:self.W, :]
        return image.transpose([2,0,1])

    def get_list(self, data_path):
        folder_list = list()
        with open(self.data_list_file, 'r') as f_index:
            reader = csv.reader(f_index)
            for row in reader:
                folder_list.append(data_path + row[0] + '/')
        return folder_list

    def non_shuffle_iterator(self):
        ''' generate n,f in scanline order '''
        f_range = range(self.frame_max - self.frame_window_size + 1)
        while True:
            for n in range(self.N_max):
                for f in f_range:
                    yield n, f

    def shuffle_iterator(self):
        ''' generate n,f randomly '''
        while True:
            n = self.rng.randint(0, self.N_max)
            f = self.rng.randint(0, self.frame_max - self.frame_window_size + 1)
            yield n,f

def generator_index(N_max, N_size, F_max, F_size=3, seed=None):
    ''' Generate index for N and F

    Args:
        N_max: video clip total size
        N_size: select size
        F_max: frame total size
        F_size: frame size = 3
        seed: None, or 0..10000
    '''    

    if seed is not None:
        rng_N_index = np.random.RandomState(seed)
        rng_F_index = np.random.RandomState(seed+1)
        while True:
            N_st = rng_N_index.randint(0, N_max-N_size+1)
            F_st = rng_F_index.randint(0, F_max-F_size+1)
            yield list(range(N_st, N_st+N_size)), list(range(F_st, F_st+F_size))
    else:
        N_st = 0
        F_st = 0
        while True:
            yield list(range(N_st, N_st+N_size)), list(range(F_st, F_st+F_size))

            F_st = (F_st+1) % (F_max-F_size+1)  # increase frame by 1
            if F_st==0:                         # increase video number by N_size
                N_st = (N_st+N_size) % (N_max-N_size+1)

def generator_dataset_5d_array(h5_file, key_list, batch_size=8, random_seed=None):
    ''' Generate 3-frame images [batch_size,3,H,W,C] 

    Args:
        h5_file: h5 file
        key_list: key selected from h5py
        batch_size: 8
        random_seed: None, or 0..10000
    '''
    with h5py.File(h5_file, 'r') as f_h5:
        N = f_h5['/N'][...].item()
        for idx_list, frame_list in generator_index(N, batch_size, F_max=7, F_size=3, seed=random_seed):
            # print(idx_list, frame_list)
            out_list = list()
            for k in key_list:
                out_list.append(f_h5[k][idx_list,:,:,:,:][:,frame_list,:,:,:])      # h5py don't allow two index list
            yield out_list

if __name__ == "__main__":
    data_path_corrupted = 'E:/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences_noise/'
    data_path_clean = 'E:/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences/'
    data_list_file = 'E:/Fall2018_Multi_warp/dataset/vimeo_septuplet/sep_trainlist.txt'
    # gen = DataGenerator(data_path, data_list_file, batch_size = 4, frame_window_size=3, shuffle=True).yield_batches()
    data = DataGenerator(data_path_corrupted, data_path_clean, data_list_file, batch_size = 4, frame_window_size=3, shuffle=False, crop=(128,128))
    genrator = data.yield_batches()

    # test data tpye
    print(genrator.__next__()[0].dtype)
    print(genrator.__next__()[-1].dtype)

    # test visualization
    fig = plt.figure()
    im = plt.imshow(np.zeros((data.H, data.W,3)), vmin=0, vmax=1)
    for frame_list in genrator:
        for count in range(data.batch_size):
            print(count)
            for frame in frame_list:
                im.set_data(frame[count,...].transpose(1,2,0))
                plt.pause(0.3)