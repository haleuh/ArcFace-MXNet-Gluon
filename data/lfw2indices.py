import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, ".."))

import argparse
import numpy as np


def read_lst(lst_path):
    lst = np.loadtxt(lst_path, dtype=str)
    return dict(zip(lst[:, -1], lst[:, 0]))


def read_lfw_pairs(pair_list_path):
    img_list1 = []
    img_list2 = []
    labels = []
    with open(pair_list_path, 'r') as f:
        f.readline()  # skip first line
        for line in f:
            parts = line.split('\t')
            if len(parts) == 3:
                img1 = parts[0] + '/' + parts[0] + '_{:04d}.jpg'.format(int(parts[1]))
                img2 = parts[0] + '/' + parts[0] + '_{:04d}.jpg'.format(int(parts[2]))
                img_list1.append(img1)
                img_list2.append(img2)
                labels.append('1')
            else:
                img1 = parts[0] + '/' + parts[0] + '_{:04d}.jpg'.format(int(parts[1]))
                img2 = parts[2] + '/' + parts[2] + '_{:04d}.jpg'.format(int(parts[3]))
                img_list1.append(img1)
                img_list2.append(img2)
                labels.append('0')
    return img_list1, img_list2, labels


def convert_lfw_pairs_to_indices(data_dir, lst_file):
    lst_path = os.path.join(data_dir, lst_file)
    lst = read_lst(lst_path)
    pair_list_path = os.path.join(data_dir, 'pairs.txt')
    f1, f2, label = read_lfw_pairs(pair_list_path)
    new_f1 = []
    new_f2 = []
    new_label = []
    for k1, k2, l in zip(f1, f2, label):
        if k1 in lst and k2 in lst:
            new_f1.append(lst[k1])
            new_f2.append(lst[k2])
            new_label.append(l)
        else:
            new_f1.append(new_f1[-1])
            new_f2.append(new_f2[-1])
            new_label.append(new_label[-1])
    new_pairs = np.stack((new_f1, new_f2, new_label), axis=1)
    new_pair_path = os.path.join(data_dir, 'lfw_pairs.txt')
    np.savetxt(new_pair_path, new_pairs, fmt='%s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', default='', help='image list')
    parser.add_argument('-i', '--input', default='', help='path to the dataset directory')
    args = parser.parse_args()
    convert_lfw_pairs_to_indices(args.input, args.list)
