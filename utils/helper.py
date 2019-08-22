import os
import logging
from datetime import datetime
from glob import glob
import mxnet as mx


def create_output_dirs(output_dir, prefix, resume=False):
    prefix_dirs = sorted(glob(os.path.join(output_dir, '{}-*'.format(prefix))), key=os.path.getmtime)
    if resume and len(prefix_dirs):
        prefix_dir = prefix_dirs[-1]
    else:
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        prefix_dir = os.path.join(output_dir, prefix + '-' + current_time)
    log_dir = os.path.join(prefix_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    ckpt_dir = os.path.join(prefix_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return log_dir, ckpt_dir


def create_logger(log_dir, prefix='log'):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = os.path.join(log_dir, '{}-{}.log'.format(prefix, current_time))
    if os.path.exists(log_file):
        os.remove(log_file)
    # Create a root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Log format
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    return root_logger


def load_params(inference, ckpt, prefix='ckpt', init=mx.initializer.Uniform(), ctx=None):
    if os.path.isfile(ckpt):
        inference.initialize(init=init, ctx=ctx)
        print('Loading checkpoint at: {}'.format(ckpt))
        inference.load_parameters(ckpt, ctx=ctx, allow_missing=False, ignore_extra=False)
        return 0
    elif os.path.isdir(ckpt):
        params = sorted(glob(os.path.join(ckpt, '{}-*.params'.format(prefix))), key=os.path.getmtime)
        if len(params):
            print('Loading checkpoint at: {}'.format(params[-1]))
            inference.load_parameters(params[-1], ctx=ctx, allow_missing=False, ignore_extra=False)
            epoch = int(os.path.basename(params[-1])[:-7].split('-')[-1])
            return epoch
    print('No checkpoint found! Randomly initialize params...')
    inference.initialize(init=init, ctx=ctx)
    return 0


def load_trainer(trainer, ckpt_dir, epoch, prefix='ckpt'):
    if epoch > 0:
        trainer_path = os.path.join(ckpt_dir, '{}-{:04d}.trainer'.format(prefix, epoch))
        if os.path.exists(trainer_path):
            print('Loading trainer at: {}'.format(trainer_path))
            trainer.load_states(trainer_path)
    else:
        print('No trainer found! Use the default trainer.')


def save_params(inference, ckpt_dir, epoch, prefix='ckpt'):
    inference.save_parameters(os.path.join(ckpt_dir, '{}-{}.params'.format(prefix, epoch)))


def save_trainer(trainer, ckpt_dir, epoch, prefix='ckpt'):
    trainer.save_states(os.path.join(ckpt_dir, '{}-{:04d}.trainer'.format(prefix, epoch)))


def print_scalars(scalars, epoch, batch, elapsed_time=None):
    out = 'E/B: {}/{}, '.format(epoch, batch)
    for k, v in scalars.items():
        out += '{}: {:.2f}, '.format(k, v)
    if elapsed_time is not None:
        out += 'Time: {:.1f} s.'.format(elapsed_time)
    else:
        out = out[:-2]
    # print(out)
    return out
