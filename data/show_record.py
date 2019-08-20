import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, ".."))

import argparse
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet.gluon.data import DataLoader, ArrayDataset
from mxnet.gluon.data.vision import ImageRecordDataset, transforms


def show_images(images, ncols=1, titles=None, wspace=0, hspace=0):
    if isinstance(images, mx.nd.NDArray):
        images = images.copy().asnumpy()
    if images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)
    images = images.astype(np.float32)
    num_images = len(images)
    nrows = int(np.ceil(num_images / float(ncols)))

    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    axes = [fig.add_subplot(nrows, ncols, i) for i in range(1, nrows * ncols + 1)]
    plt.setp(axes, xticks=[], yticks=[])

    for i, image in enumerate(images):
        axes[i].axis('off')
        if image.min() < 0:
            image = (image * 0.5) + 0.5
        axes[i].imshow(image)
        if titles is not None:
            axes[i].set_title(titles[i])
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()


def display():
    transform = transforms.ToTensor()
    dataset = ImageRecordDataset(args.input).transform_first(transform)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=args.shuffle, last_batch='keep',
                        num_workers=args.num_workers, pin_memory=False)
    for idx, (img, label) in enumerate(loader):
        show_images(img, ncols=min(8, args.batch_size))
        # print(label)
        input('Press Enter...')


def display_multi():
    transform = transforms.ToTensor()
    dataset = []
    for rec in args.inputs:
        dataset.append(ImageRecordDataset(rec).transform_first(transform))
    dataset = ArrayDataset(*dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=args.shuffle, last_batch='keep',
                        num_workers=args.num_workers, pin_memory=False)
    for idx, batch_data in enumerate(loader):
        batch_img = []
        for (img, _) in batch_data:
            batch_img.append(img)
        batch_img = mx.nd.concat(*batch_img, dim=0)
        show_images(batch_img, ncols=min(8, args.batch_size))
        input('Press Enter...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1987, type=int, help='manual random seed, -1 to ignore')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--shuffle', action='store_true', help='shuffle the dataset')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-i', '--input', default='', help='path to dataset record')
    parser.add_argument('--multi', action='store_true', help='show array dataset')
    args = parser.parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)
        mx.random.seed(args.seed)

    if args.multi:
        base_dir = os.path.dirname(args.input)
        args.inputs = sorted(glob(os.path.join(base_dir, '*.rec')))
        print(args.inputs)
        display_multi()
    else:
        display()
