import numpy as np
import h5py

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
    for out_list in generator_dataset_5d_array('../dataset_h5/train_data.h5', ['/im'], batch_size=2, random_seed=100):
        print(out_list[0].shape)