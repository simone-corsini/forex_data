import h5py
import numpy as np
import argparse
import os
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare sanity train data', add_help=True)
    parser.add_argument('--datafile', type=str, help='Path to the datafile')
    parser.add_argument('--sanity_train', type=int, help='Number of samples per class in the sanity train set')
    parser.add_argument('--sanity_val', type=int, help='Number of samples per class in the sanity val set')
    parser.add_argument('--targets', type=int, help='Number of targets')

    args = parser.parse_args()

    sanity_datafile = args.datafile.split('.')[0] + '_sanity.h5'

    labels = [i for i in range(args.targets)]

    label_selected = defaultdict(list)
    for label in labels:
        label_selected[label] = 0

    X_train_sanity = []
    y_train_sanity = []

    X_val_sanity = []
    y_val_sanity = [] 

    with h5py.File(args.datafile, 'r') as file:
        X_train = file['X_train']
        y_train = file['y_train']

        X_val = file['X_val']
        y_val = file['y_val']


        x_train_indexes = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)

        for i in x_train_indexes:
            if (label_selected[y_train[i]] < args.sanity_train):
                label_selected[y_train[i]] += 1
                X_train_sanity.append(X_train[i])
                y_train_sanity.append(y_train[i])

        for label in labels:
            label_selected[label] = 0

        x_val_indexes = np.random.choice(X_val.shape[0], X_val.shape[0], replace=False)

        for i in x_val_indexes:
            if (label_selected[y_val[i]] < args.sanity_val):
                label_selected[y_val[i]] += 1
                X_val_sanity.append(X_val[i])
                y_val_sanity.append(y_val[i])



    print('Sanity train set size: ', len(X_train_sanity))
    print('Sanity val set size: ', len(X_val_sanity))

    with h5py.File(sanity_datafile, 'w') as file:
        file.create_dataset('X_train', data=X_train_sanity)
        file.create_dataset('y_train', data=y_train_sanity)
        file.create_dataset('X_val', data=X_val_sanity)
        file.create_dataset('y_val', data=y_val_sanity)

        print(file['X_train'].shape)
        print(file['y_train'].shape)
        print(file['X_val'].shape)
        print(file['y_val'].shape)

        