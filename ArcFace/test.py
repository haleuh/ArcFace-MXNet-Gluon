import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, ".."))

import argparse
import timeit
from collections import OrderedDict
import numpy as np

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import ImageRecordDataset

from utils import helper
from utils.evaluation import evaluate_pairs
from ArcFace.resnet import resnet100
from ArcFace.transforms import ToTensor


def extract_features(inference, test_loader, ctx):
    features = []
    labels = []
    for _, (test_image, test_label) in enumerate(test_loader):
        images = gluon.utils.split_and_load(test_image, ctx, even_split=False)
        if test_label.ndim > 1:
            test_label = test_label[:, 0]
        labels.append(test_label.asnumpy())
        with autograd.predict_mode():
            for x in images:
                embedding = inference(x)
                embedding = mx.nd.L2Normalization(embedding, mode='instance')
                # flip image
                embedding_flip = inference(x.flip(axis=3))
                embedding_flip = mx.nd.L2Normalization(embedding_flip, mode='instance')
                embedding = embedding + embedding_flip
                embedding = mx.nd.L2Normalization(embedding, mode='instance')
                features.append(embedding.asnumpy())
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return features, labels


def eval_lfw(inference, test_rec, test_loader, ctx):
    features, _ = extract_features(inference, test_loader, ctx)
    pair_file = os.path.join(os.path.dirname(test_rec), 'lfw_pairs.txt')
    mu, std, t, accuracies = evaluate_pairs(pair_file, features)
    return mu, std, t, accuracies


def evaluate():
    # Datasets
    test_transform = ToTensor()
    test_dataset = ImageRecordDataset(args.test_rec).transform_first(test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, last_batch='keep',
                             num_workers=args.num_workers, pin_memory=True)

    logger.info('Evaluate model {}'.format(os.path.basename(args.model)))
    if args.model.endswith('.params'):
        # Create inference
        inference = resnet100(args.num_classes, emb_size=args.emb_size,
                              s=args.margin_s, a=args.margin_a, m=args.margin_m, b=args.margin_b)
        inference.hybridize(static_alloc=True, static_shape=True)
        helper.load_params(inference, args.model, ctx=ctx)
        inference = inference.features
    elif args.model.endswith('-symbol.json'):
        # Load model symbol and params
        sym = mx.sym.load_json(open(args.model, 'r').read())
        inference = gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var('data'))
        inference.load_parameters(args.model[:-11] + '0000.params', ctx=ctx)
    else:
        print('Incorrect model: {}'.format(args.model))
        return

    # Test LFW
    if args.test_name.lower() == 'lfw':
        print('Evaluating LFW...')
        start_time = timeit.default_timer()
        mu, std, t, accuracies = eval_lfw(inference, args.test_rec, test_loader, ctx)
        elapsed_time = timeit.default_timer() - start_time
        scout = helper.print_scalars(OrderedDict([('mu', mu), ('std', std), ('t', t)]), 0, 0, elapsed_time)
        logger.info(scout)
        accuracies = accuracies.tolist() + [mu, std]
        logger.info(' '.join('{:.2f}'.format(x) for x in accuracies))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--gpus', default='0', help='gpu devices')
    parser.add_argument('--num_workers', default=4, type=int, help='gpu devices')
    # output
    parser.add_argument('--output_dir', default='runs', help='output directory')
    parser.add_argument('--prefix', default='arcface', help='prefix')
    parser.add_argument('--model', default='shared/ArcFace/arcface-glint-nocolor-symbol.json',
                        help='pretrained model')
    parser.add_argument('--test_name', default='lfw', help='name of the test set')
    parser.add_argument('--test_rec', default='/mnt/Datasets/lfw/lfw_dlib_prnet.rec', help='test record')

    parser.add_argument('--emb_size', default=512, type=int, help='embedding size')
    parser.add_argument('--num_classes', default=180855, type=int, help='number of classes')
    parser.add_argument('--margin_s', default=64.0, type=float, help='scale for feature')
    parser.add_argument('--margin_a', default=1.0, type=float, help='margin for sphereface loss')
    parser.add_argument('--margin_m', default=0.3, type=float, help='margin for arcface loss')
    parser.add_argument('--margin_b', default=0.2, type=float, help='margin for cosineface loss')

    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    args = parser.parse_args()

    # Output directories
    args.log_dir, args.ckpt_dir = helper.create_output_dirs(args.output_dir, args.prefix)
    logger = helper.create_logger(args.log_dir, args.prefix)
    logger.info(args)

    ctx = [mx.gpu(int(gpu_id)) for gpu_id in args.gpus.split(',')] if args.gpus != '-1' else [mx.cpu()]
    args.batch_size = args.batch_size * len(ctx)

    evaluate()
