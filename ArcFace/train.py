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
from mxnet.gluon.data.vision import ImageRecordDataset, transforms

from utils import helper
from ArcFace.resnet import resnet100
from ArcFace.transforms import ToTensor
from ArcFace.test import test_lfw


def train():
    # Create inference
    inference = resnet100(args.num_classes, emb_size=args.emb_size,
                          s=args.margin_s, a=args.margin_a, m=args.margin_m, b=args.margin_b)
    # Load inference params
    if args.init.lower() == 'xavier':
        init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
    else:
        init = mx.initializer.Uniform()
    if args.model:
        helper.load_params(inference, args.model, ctx=ctx)
        cur_iter = 0
    else:
        cur_iter = helper.load_params(inference, args.ckpt_dir, prefix=args.prefix, init=init, ctx=ctx)
    # Hybrid mode --> Symbol mode
    inference.hybridize(static_alloc=True, static_shape=True)

    # Datasets
    if args.color:
        train_transform = transforms.Compose([transforms.RandomFlipLeftRight(),
                                              transforms.RandomColorJitter(0.1, 0.1, 0.1),
                                              ToTensor()])
    else:
        train_transform = transforms.Compose([transforms.RandomFlipLeftRight(),
                                              ToTensor()])

    train_dataset = ImageRecordDataset(args.train_rec).transform_first(train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, last_batch='discard',
                              num_workers=args.num_workers, pin_memory=True)
    test_transform = ToTensor()
    test_dataset = ImageRecordDataset(args.test_rec).transform_first(test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, last_batch='keep',
                             num_workers=args.num_workers, pin_memory=False)

    # Create learning rate scheduler
    iterations_per_epoch = int(len(train_dataset) / args.batch_size)
    lr_steps = [s * iterations_per_epoch for s in args.lr_steps]
    print('Learning rate drops after iterations: {}'.format(lr_steps))
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=lr_steps, factor=0.1)

    # Create trainer
    trainer = gluon.Trainer(inference.collect_params(), optimizer='sgd',
                            optimizer_params={'learning_rate': args.lr,
                                              'wd': args.wd,
                                              'lr_scheduler': lr_scheduler,
                                              'rescale_grad': 1. / len(ctx)})
    # Load trainer from saved states
    helper.load_trainer(trainer, args.ckpt_dir, cur_iter, prefix=args.prefix)

    # Define loss functions
    softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    # Define metric losses
    metric_ce_loss = mx.metric.Loss('CE-Loss')
    best_acc = 80  # only save the model if the accuracy is better than 80%
    # Start training
    print('Start to train {}...'.format(args.prefix))
    start_epoch = cur_iter // iterations_per_epoch
    for cur_epoch in range(start_epoch + 1, args.max_epoch + 1):
        start_time = timeit.default_timer()
        for batch_idx, (image, label) in enumerate(train_loader):
            if label.ndim > 1:
                label = label[:, 0]  # skip the landmarks
            # if batch_idx > 0: break
            cur_iter += 1
            images = gluon.utils.split_and_load(image, ctx)
            labels = gluon.utils.split_and_load(label, ctx)
            with autograd.record(train_mode=True):
                losses = []
                for x, y in zip(images, labels):
                    fc = inference(x, y)
                    loss_ce = softmax_cross_entropy(fc, y)
                    losses.append(loss_ce)
                    # update metrics
                    metric_ce_loss.update(None, preds=loss_ce)
                for l in losses:
                    l.backward()
            trainer.step(image.shape[0])

            if (batch_idx % args.log_interval == 0) or (batch_idx == iterations_per_epoch - 1):
                elapsed_time = timeit.default_timer() - start_time
                scout = helper.print_scalars(OrderedDict([metric_ce_loss.get()]),
                                             cur_epoch, batch_idx, elapsed_time)
                if batch_idx == iterations_per_epoch - 1:
                    logger.info(scout)
                else:
                    print(scout)
                start_time = timeit.default_timer()
                metric_ce_loss.reset()

            if (batch_idx % args.test_interval == 0) or (batch_idx == iterations_per_epoch - 1):
                # if batch_idx > 0: break
                start_time = timeit.default_timer()
                mu, std, t = test_lfw(inference.features, args.test_rec, test_loader, ctx)
                elapsed_time = timeit.default_timer() - start_time
                if mu >= best_acc:
                    best_acc = mu
                    # Save trained model
                    logger.info('Find better model at epoch {}, batch {}'.format(cur_epoch, batch_idx))
                    helper.save_params(inference, args.ckpt_dir, cur_iter, prefix=args.prefix + '-best')
                scout = helper.print_scalars(OrderedDict([('mu', mu), ('std', std), ('t', t)]),
                                             cur_epoch, batch_idx, elapsed_time)
                logger.info(scout)

        # Save trained model
        helper.save_params(inference, args.ckpt_dir, cur_iter, prefix=args.prefix)
        helper.save_trainer(trainer, args.ckpt_dir, cur_iter, prefix=args.prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument('--seed', default=1987, type=int, help='manual random seed, -1 to ignore')
    parser.add_argument('--gpus', default='0', help='gpu devices')
    parser.add_argument('--num_workers', default=4, type=int, help='number of parallel workers')
    # output
    parser.add_argument('--output_dir', default='runs', help='output directory')
    parser.add_argument('--prefix', default='arcface', help='prefix')
    parser.add_argument('--model', default='', help='pretrained model')
    # train
    parser.add_argument('--train_rec', default='/mnt/Datasets/Glint/glint_dlib_prnet.rec', help='train record')
    parser.add_argument('--test_rec', default='/mnt/Datasets/lfw/lfw_dlib_prnet.rec', help='test record')

    parser.add_argument('--emb_size', default=512, type=int, help='embedding size')
    parser.add_argument('--num_classes', default=180855, type=int, help='number of classes')
    parser.add_argument('--margin_s', default=64.0, type=float, help='scale for feature')
    parser.add_argument('--margin_a', default=1.0, type=float, help='margin for sphereface loss')
    parser.add_argument('--margin_m', default=0.3, type=float, help='margin for arcface loss')
    parser.add_argument('--margin_b', default=0.2, type=float, help='margin for cosineface loss')

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--max_epoch', default=30, type=int, help='maximum number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--color', action='store_true', help='color jitter')
    parser.add_argument('--init', default='xavier', help='initializer')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_steps', default='9,12,14', help='learning rate decay steps')

    parser.add_argument('--log_interval', default=100, type=int, help='log interval')
    parser.add_argument('--test_interval', default=1000, type=int, help='test interval')

    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    # Pre-process arguments
    args.lr_steps = list(map(int, args.lr_steps.split(',')))
    # Output directories
    args.log_dir, args.ckpt_dir = helper.create_output_dirs(args.output_dir, args.prefix, args.resume)
    logger = helper.create_logger(args.log_dir, args.prefix)
    logger.info(args)

    if args.seed >= 0:
        np.random.seed(args.seed)
        mx.random.seed(args.seed)

    ctx = [mx.gpu(int(gpu_id)) for gpu_id in args.gpus.split(',')] if args.gpus != '-1' else [mx.cpu()]
    args.batch_size = args.batch_size * len(ctx)
    # Main function
    train()
