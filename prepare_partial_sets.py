import h5py
import numpy as np
import argparse
import os
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare partial train data', add_help=True)
    parser.add_argument('--datafile', type=str, help='Path to the datafile', required=True)
    parser.add_argument('--partial_train', type=int, help='Number of samples per class in the partial train set', required=True)
    parser.add_argument('--partial_val', type=int, help='Number of samples per class in the partial val set', default=0)
    parser.add_argument('--partial_test', type=int, help='Number of samples per class in the partial test set', default=0)
    parser.add_argument('--targets', type=int, help='Number of targets', required=True )

    args = parser.parse_args()

    partial_datafile = args.datafile.split('.')[0] + f'_train_{args.partial_train}_val_{args.partial_val}_test_{args.partial_test}.h5'


    labels = [i for i in range(args.targets)]

    label_selected = defaultdict(int)
    for label in labels:
        label_selected[label] = 0

    X_train_partial = []
    y_train_partial = []

    X_val_partial = []
    y_val_partial = [] 

    X_test_partial = []
    y_test_partial = [] 

    with h5py.File(args.datafile, 'r') as file:
        X_train = file['X_train']
        y_train = file['y_train']

        X_val = file['X_val']
        y_val = file['y_val']

        X_test = file['X_test']
        y_test = file['y_test']

        for i in range(X_train.shape[0]):
            if (label_selected[y_train[i]] < args.partial_train):
                label_selected[y_train[i]] += 1
                X_train_partial.append(X_train[i])
                y_train_partial.append(y_train[i])
            elif not any(value < args.partial_train for value in label_selected.values()):
                break

        for label in labels:
            label_selected[label] = 0

        for i in range(X_val.shape[0]):
            if (label_selected[y_val[i]] < args.partial_val):
                label_selected[y_val[i]] += 1
                X_val_partial.append(X_val[i])
                y_val_partial.append(y_val[i])
            elif not any(value < args.partial_val for value in label_selected.values()):
                break

        for label in labels:
            label_selected[label] = 0

        for i in range(X_test.shape[0]):
            if (label_selected[y_test[i]] < args.partial_test):
                label_selected[y_test[i]] += 1
                X_val_partial.append(X_test[i])
                y_val_partial.append(y_test[i])
            elif not any(value < args.partial_test for value in label_selected.values()):
                break


    print('partial train set size: ', len(X_train_partial))
    print('partial val set size: ', len(X_val_partial))
    print('partial test set size: ', len(X_test_partial))

    with h5py.File(partial_datafile, 'w') as file:
        file.create_dataset('X_train', data=X_train_partial)
        file.create_dataset('y_train', data=y_train_partial)
        file.create_dataset('X_val', data=X_val_partial)
        file.create_dataset('y_val', data=y_val_partial)
        file.create_dataset('X_test', data=X_test_partial)
        file.create_dataset('y_test', data=y_test_partial)

        print('Partial X_train set shape: ', file['X_train'].shape)
        print('Partial y_train set shape: ', file['y_train'].shape)
        print('Partial X_val set shape: ', file['X_val'].shape)
        print('Partial y_val set shape: ', file['y_val'].shape)
        print('Partial X_test set shape: ', file['X_test'].shape)
        print('Partial y_test set shape: ', file['y_test'].shape)

        