import h5py
import numpy as np

np.set_printoptions(threshold=np.inf)

with h5py.File('data/train_sets/set_o60_f10_t2_max.h5', 'r') as file:
    X_train = file['X_train']
    
    print(X_train.attrs['min'])
    print(X_train.attrs['max'])
    print(X_train[0])

    # print(X_train[15])
    # print(y_train[15])
    

