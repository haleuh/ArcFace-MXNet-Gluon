import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, ".."))

import argparse
import math
import tensorflow as tf
from PRNet.PRN import PRN
from data import read_list, write_list
import timeit


def read_index(idx_path):
    if os.path.exists(idx_path):
        with open(idx_path, 'r') as f:
            return int(f.readline())
    return 0


def write_index(idx_path, idx):
    with open(idx_path, 'w') as f:
        f.write(str(idx))


def main():
    # Read image list
    idx = read_index(args.index_path)
    image_list = list(read_list(args.input))[idx:]
    image_only_list = [item[1] for item in image_list]
    bbox_only_list = [item[3:7] for item in image_list]
    data_list = (image_only_list, bbox_only_list)
    # ---- init PRN
    prn = PRN(args.dataset, data_list, args.batch_size)
    batch_idx = 0
    start = 0
    num_batch = math.ceil(len(image_list) / args.batch_size) + 1
    start_time = timeit.default_timer()
    try:
        while True:
            if (batch_idx + 1) % 100 == 0:
                elapsed_time = timeit.default_timer() - start_time
                remaining_time = (elapsed_time / (batch_idx + 1)) * (num_batch - batch_idx - 1) / 60
                print('Processing batch: {}/{}\tRemaining time:{:.2f} mins'.format(batch_idx + 1,
                                                                                   num_batch, remaining_time))
            kpts = prn.process_next()
            end = start + len(kpts)
            items = image_list[start:end]
            new_image_list = [item[:7] + kpt.tolist() + item[7:] for (item, kpt) in zip(items, kpts)]
            write_list(args.output, new_image_list, mode='a')
            start = end
            batch_idx += 1
            idx += len(kpts)
            write_index(args.index_path, idx)
    except tf.errors.OutOfRangeError:
        os.remove(args.index_path)
        print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-d', '--dataset', default='', type=str,
                        help='path to the input directory, where input images are stored')
    parser.add_argument('-i', '--input', default='', type=str,
                        help='path to the input .lst file')
    parser.add_argument('-o', '--output', default='', type=str,
                        help='path to the output .lst file')
    parser.add_argument('--index_path', default='', type=str,
                        help='path to the index file')
    parser.add_argument('--gpu', default='0', type=str, help='set gpu id')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # GPU number

    if not args.output:
        args.output = args.input[:-4] + '_prnet.lst'
    if not args.index_path:
        args.index_path = os.path.join(os.path.dirname(args.input), 'index.txt')
    main()
