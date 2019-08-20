import os
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from utils.evaluation import evaluate_pairs


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


def test_lfw(inference, test_rec, test_loader, ctx):
    features, _ = extract_features(inference, test_loader, ctx)
    pair_file = os.path.join(os.path.dirname(test_rec), 'lfw_pairs.txt')
    mu, std, t, _ = evaluate_pairs(pair_file, features)
    return mu, std, t
