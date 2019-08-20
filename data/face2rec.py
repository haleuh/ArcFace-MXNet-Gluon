#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, ".."))

import mxnet as mx
import argparse
import cv2
import numpy as np
import time
import traceback

from data import read_list
from data.face_alignment import get_landmarks, align_face, align_landmarks

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


def image_encode(args, i, item, q_out):
    """Reads, preprocesses, packs the image and put it back in output queue.
    Parameters
    ----------
    args
    i: int
    item: list
    q_out: Queue
    """
    # adding suffix to image name
    name, ext = os.path.splitext(item[1])
    new_name = name + args.suffix + ext
    fullpath = os.path.join(args.root, new_name)

    if args.pack_label:
        landmarks = np.array(item[7:143]).reshape(68, 2)
        lms = align_landmarks(landmarks)
        lms = get_landmarks(lms).flatten().tolist()
        header = mx.recordio.IRHeader(0, item[2:3] + lms, item[0], 0)
    else:
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)

    if args.pass_through:
        try:
            with open(fullpath, 'rb') as fin:
                img = fin.read()
            s = mx.recordio.pack(header, img)
            q_out.put((i, s, item))
        except Exception as e:
            traceback.print_exc()
            print('pack_img error:', item[1], e)
            q_out.put((i, None, item))
        return

    try:
        img = cv2.imread(fullpath, args.color)
    except:
        traceback.print_exc()
        print('imread error trying to load file: %s ' % fullpath)
        q_out.put((i, None, item))
        return
    if img is None:
        print('imread read blank (None) image for file: %s' % fullpath)
        q_out.put((i, None, item))
        return

    # Align face
    landmark = np.array(item[7:143]).reshape(68, 2)
    img = align_face(img, landmark)
    try:
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
        q_out.put((i, s, item))
    except Exception as e:
        traceback.print_exc()
        print('pack_img error on file: %s' % fullpath, e)
        q_out.put((i, None, item))
        return


def read_worker(args, q_in, q_out):
    """Function that will be spawned to fetch the image
    from the input queue and put it back to output queue.
    Parameters
    ----------
    args: object
    q_in: queue
    q_out: queue
    """
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)


def write_worker(q_out, fname, working_dir):
    """Function that will be spawned to fetch processed image
    from the output queue and write to the .rec file.
    Parameters
    ----------
    q_out: queue
    fname: string
    working_dir: string
    """
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)

    fname_rec = os.path.splitext(fname)[0] + '{}.rec'.format(args.suffix)
    fname_idx = os.path.splitext(fname)[0] + '{}.idx'.format(args.suffix)

    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)

            if count % 1000 == 0:
                cur_time = time.time()
                print('time: {:.2f} count: {}'.format(cur_time - pre_time, count))
                pre_time = cur_time
            count += 1


def parse_args():
    """Defines all arguments.
    Returns
    -------
    args object that contains all the params
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('prefix', help='prefix of input/output lst and rec files.')
    parser.add_argument('root', help='path to folder containing images.')
    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--pass_through', action='store_true',
                        help='whether to skip transformation and save image as is')
    rgroup.add_argument('--resize', type=int, default=112,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--suffix', type=str, default='',
                        help='add suffix for image filename')
    rgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    rgroup.add_argument('--pack-label', action='store_true',
                        help='Whether to also pack multi dimensional label in the record file')
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args


if __name__ == '__main__':
    args = parse_args()
    if os.path.isdir(args.prefix):
        working_dir = args.prefix
    else:
        working_dir = os.path.dirname(args.prefix)
    files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
             if os.path.isfile(os.path.join(working_dir, fname))]
    count = 0
    for fname in files:
        if fname.startswith(args.prefix) and fname.endswith('.lst'):
            print('Creating .rec file from', fname, 'in', working_dir)
            count += 1
            image_list = read_list(fname)
            # -- write_record -- #
            if args.num_thread > 1 and multiprocessing is not None:
                q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
                q_out = multiprocessing.Queue(1024)
                # define the process
                read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out))
                                for i in range(args.num_thread)]
                # process images with num_thread process
                for p in read_process:
                    p.start()
                # only use one process to write .rec to avoid race-condition
                write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, working_dir))
                write_process.start()
                # put the image list into input queue
                for i, item in enumerate(image_list):
                    q_in[i % len(q_in)].put((i, item))
                for q in q_in:
                    q.put(None)
                for p in read_process:
                    p.join()

                q_out.put(None)
                write_process.join()
            else:
                print('multiprocessing not available, fall back to single threaded encoding')
                try:
                    import Queue as queue
                except ImportError:
                    import queue
                q_out = queue.Queue()
                fname = os.path.basename(fname)
                fname_rec = os.path.splitext(fname)[0] + '{}.rec'.format(args.suffix)
                fname_idx = os.path.splitext(fname)[0] + '{}.idx'.format(args.suffix)

                record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                                       os.path.join(working_dir, fname_rec), 'w')
                cnt = 0
                pre_time = time.time()
                for i, item in enumerate(image_list):
                    image_encode(args, i, item, q_out)
                    if q_out.empty():
                        continue
                    _, s, _ = q_out.get()
                    record.write_idx(item[0], s)
                    if cnt % 1000 == 0:
                        cur_time = time.time()
                        print('time: {:.2f} count: {}'.format(cur_time - pre_time, cnt))
                        pre_time = cur_time
                    cnt += 1
    if not count:
        print('Did not find and list file with prefix %s' % args.prefix)
