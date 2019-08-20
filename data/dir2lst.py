import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, ".."))

import argparse
from data import write_list


def list_image(root, exts=('.jpg', '.jpeg', '.png', '.bmp')):
    i = 0
    cat = {}
    for path, dirs, files in os.walk(root, followlinks=True):
        dirs.sort()
        files.sort()
        for fname in files:
            fpath = os.path.join(path, fname)
            ext = os.path.splitext(fpath)[1]
            if os.path.isfile(fpath) and ext in exts:
                if path not in cat:
                    cat[path] = len(cat)
                out = [i, os.path.relpath(fpath, root), cat[path]]
                yield out
                i += 1
                if i % 1000 == 0:
                    print('Listed {} images'.format(i))


def main():
    image_list = list_image(args.input)
    write_list(args.output, image_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a list of images and ids for a dataset')

    parser.add_argument('-i', '--input', default='', type=str,
                        help='path to the input directory, where input images are stored')
    parser.add_argument('-o', '--output', default='', type=str,
                        help='path to the output .lst file')
    args = parser.parse_args()
    main()
