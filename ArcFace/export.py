import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, ".."))

import argparse
import mxnet as mx
from utils import helper
from ArcFace.resnet import resnet100


def export():
    # Create inference
    inference = resnet100(args.num_classes, emb_size=args.emb_size)
    inference.hybridize(static_alloc=True, static_shape=True)

    # Load inference params
    helper.load_params(inference, args.model)

    # Export model
    x = mx.nd.ones(shape=args.img_size)
    inference.features.hybridize(static_alloc=True, static_shape=True)
    inference.features.forward(x)
    inference.features.export(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # output
    parser.add_argument('--output', default='shared/ArcFace/arcface-glint-nocolor', help='output model')
    parser.add_argument('--model', default='shared/ArcFace/arcface-glint-nocolor-best-337551.params',
                        help='pretrained model')
    parser.add_argument('--img_size', default='112,112', type=str, help='image size')
    parser.add_argument('--emb_size', default=512, type=int, help='embedding size')
    parser.add_argument('--num_classes', default=180855, type=int, help='number of classes')

    args = parser.parse_args()
    args.img_size = [1, 3] + list(map(int, args.img_size.split(',')))

    export()
