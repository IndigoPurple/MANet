import numpy as np
import h5py

def generator_index(N, K=1, seed=None):
    '''
        random_seed: None, or 0..10000
    '''    

    if seed is not None:
        rng_index = np.random.RandomState(seed)
        while True:
            s = rng_index.randint(0,N-K)
            yield list(range(s, s+K))
    else:
        s = -K
        while True:
            s = (s+K) % (N-K+1)
            yield list(range(s, s+K))

def generator_dataset(h5_file, key_list, batch_size=8, random_seed=None):
    with h5py.File(h5_file, 'r') as f_h5:
        N = f_h5['/N'][...].item()
        for idx_list in generator_index(N, K=batch_size, seed=random_seed):
            print('{} out of {}'.format(idx_list,N))
            out_list = list()
            for k in key_list:
                out_list.append(f_h5[k][idx_list,...])
            yield out_list

if __name__ == "__main__":
    for out_list in generator_dataset('train_data.h5', ['/im1', '/im2'], batch_size=10, random_seed=None):
        print(out_list[0].shape)