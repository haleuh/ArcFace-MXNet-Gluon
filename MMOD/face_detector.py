import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, ".."))

import argparse
import numpy as np
import cv2
import dlib
from multiprocessing import Pool
from data import read_list, write_list


# parallel processing
def progress_bar(data, func, num_processes=1):
    if num_processes == 1:
        for i, x in enumerate(map(func, data), 1):
            print('Done {0:%}'.format(i / len(data)), end='\r')
            yield x
    else:
        with Pool(num_processes) as p:
            for i, x in enumerate(p.imap(func, data), 1):
                print('Done {0:%}'.format(i / len(data)), end='\r')
                yield x


def detect_face(image_list):
    dlib_detector = dlib.cnn_face_detection_model_v1(args.face_model)
    face_list = []
    for item in image_list:
        img = cv2.imread(os.path.join(args.dataset, item[1]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            max_size = max(img.shape[0], img.shape[1])
            if max_size > 512:
                scale = 512. / max_size
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            else:
                scale = 1.
            dets = dlib_detector(img, 1)
            img_center = np.array(img.shape[:2]) / 2
            if len(dets) > 0:
                max_idx = 0
                max_cri = 0
                if len(dets) > 1:
                    # Find the largest bounding box and closest to the image center
                    for i, det in enumerate(dets):
                        det_center = np.array([(det.rect.top() + det.rect.bottom()) / 2,
                                               (det.rect.left() + det.rect.right()) / 2])
                        dist_center = np.sum(np.power(det_center - img_center, 2))
                        det_cri = det.rect.width() * det.rect.height() - 2 * dist_center
                        if det_cri > max_cri:
                            max_idx = i
                            max_cri = det_cri
                # if max_idx > 0:
                #     print(item[1])
                # Write bounding box
                bbox = np.array([dets[max_idx].rect.left(), dets[max_idx].rect.top(),
                                 dets[max_idx].rect.right(), dets[max_idx].rect.bottom()], dtype=np.float32) / scale
                bbox = bbox.astype(np.int32)
                item = item[:3] + bbox.tolist() + item[3:]
                face_list.append(item)
    return face_list


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main():
    image_list = read_list(args.input)
    image_list = list(image_list)
    if len(image_list) >= 1e6:
        image_list_chunks = list(split(image_list, 100))[args.skip:]
        for i, image_list_chunk in enumerate(image_list_chunks):
            print('Processing chunk: {}'.format(i + args.skip))
            image_list_chunk_chunks = list(split(image_list_chunk, 100))
            face_list_chunks = progress_bar(image_list_chunk_chunks, detect_face, args.num_processes)
            face_list = [item for sublist in face_list_chunks for item in sublist]
            write_list(args.output, face_list, mode='a')
    else:
        image_list_chunks = list(split(image_list, 100))
        face_list_chunks = progress_bar(image_list_chunks, detect_face, args.num_processes)
        face_list = [item for sublist in face_list_chunks for item in sublist]
        write_list(args.output, face_list, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--num_processes', default=1, type=int, help='number of parallel processes')
    parser.add_argument('-f', '--face_model', default='shared/MMOD/mmod_human_face_detector.dat',
                        help='dlib face detection model')
    parser.add_argument('-d', '--dataset', default='', type=str,
                        help='path to the input directory, where input images are stored')
    parser.add_argument('-i', '--input', default='', type=str,
                        help='path to the input .lst file')
    parser.add_argument('-o', '--output', default='', type=str,
                        help='path to the output .lst file')
    parser.add_argument('-s', '--skip', default=0, type=int,
                        help='skip first number of chunks')

    args = parser.parse_args()
    if not args.output:
        args.output = args.input[:-4] + '_dlib.lst'
    main()
