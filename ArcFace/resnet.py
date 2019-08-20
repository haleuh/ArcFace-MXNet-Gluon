import math
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock


class ArcFace(HybridBlock):
    def __init__(self, num_classes, emb_size=512, s=64., m=0.5, wd_mult=10, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi - m) * m
        self.t = math.cos(math.pi - m)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(num_classes, emb_size),
                                          init=None, dtype='float32',
                                          wd_mult=wd_mult,
                                          allow_deferred_init=True)

    def hybrid_forward(self, F, x, y, weight):
        weight = F.L2Normalization(weight, mode='instance')
        x = F.L2Normalization(x, mode='instance') * self.s
        fc = F.FullyConnected(x, weight, no_bias=True, num_hidden=self.num_classes)
        zy = F.pick(fc, y, axis=1)
        cos_t = zy / self.s
        cond_v = cos_t - self.t
        cond = F.Activation(data=cond_v, act_type='relu')
        body = cos_t * cos_t
        body = 1.0 - body
        sin_t = F.sqrt(body)
        new_zy = cos_t * self.cos_m
        b = sin_t * self.sin_m
        new_zy = (new_zy - b) * self.s
        zy_keep = zy - self.s * self.mm
        new_zy = F.where(cond, new_zy, zy_keep)
        diff = new_zy - zy
        diff = F.expand_dims(diff, 1)
        y_one_hot = F.one_hot(y, depth=self.num_classes, on_value=1.0, off_value=0.0)
        body = F.broadcast_mul(y_one_hot, diff)
        fc = fc + body
        return fc


class CombineFace(HybridBlock):
    def __init__(self, num_classes, emb_size=512, s=64., a=1., m=0.3, b=0.2, **kwargs):
        super(CombineFace, self).__init__(**kwargs)
        assert m != 0.
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.a = a
        self.b = b
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(num_classes, emb_size),
                                          init=None, dtype='float32',
                                          allow_deferred_init=True)

    def hybrid_forward(self, F, x, y, weight):
        weight = F.L2Normalization(weight, mode='instance')
        x = F.L2Normalization(x, mode='instance') * self.s
        fc = F.FullyConnected(x, weight, no_bias=True, num_hidden=self.num_classes)
        zy = F.pick(fc, y, axis=1)
        cos_t = zy / self.s
        t = F.arccos(cos_t)
        t = t * self.a + self.m
        body = F.cos(t)
        body = body - self.b
        new_zy = body * self.s
        diff = new_zy - zy
        diff = F.expand_dims(diff, 1)
        y_one_hot = F.one_hot(y, depth=self.num_classes, on_value=1.0, off_value=0.0)
        body = F.broadcast_mul(y_one_hot, diff)
        fc = fc + body
        return fc


class EmbeddingBlock(nn.HybridBlock):
    def __init__(self, emb_size=512, **kwargs):
        super(EmbeddingBlock, self).__init__(**kwargs)
        self.out = nn.HybridSequential(prefix='')
        with self.name_scope():
            self.out.add(nn.BatchNorm(scale=True, epsilon=2e-5))
            self.out.add(nn.Dropout(rate=0.4))
            self.out.add(nn.Dense(emb_size))
            self.out.add(nn.BatchNorm(scale=False, epsilon=2e-5))

    def hybrid_forward(self, F, x):
        x = self.out(x)
        return x


class BasicBlock(HybridBlock):
    def __init__(self, channels, stride, downsample=False, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm(scale=True, epsilon=2e-5)
        self.conv1 = nn.Conv2D(channels, 3, 1, 1, use_bias=False)
        self.bn2 = nn.BatchNorm(scale=True, epsilon=2e-5)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2D(channels, 3, stride, 1, use_bias=False)
        self.bn3 = nn.BatchNorm(scale=True, epsilon=2e-5)
        if downsample:
            self.conv_down = nn.Conv2D(channels, 3, stride, 1, use_bias=False)
            self.bn_down = nn.BatchNorm(scale=True, epsilon=2e-5)
        else:
            self.conv_down = None
            self.bn_down = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn3(x)
        if self.conv_down and self.bn_down:
            residual = self.conv_down(residual)
            residual = self.bn_down(residual)
        return x + residual


class ResNet(HybridBlock):
    def __init__(self, block, layers, channels, num_classes, emb_size=512,
                 s=64., a=1., m=0.3, b=0.2, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
            self.features.add(nn.BatchNorm(scale=True, epsilon=2e-5))
            self.features.add(nn.PReLU())

            for i, num_layer in enumerate(layers):
                self.features.add(self._make_layer(block, num_layer, channels[i+1], i+1))

            self.features.add(EmbeddingBlock(emb_size))
            if a == 1. and b == 0.:
                self.output = ArcFace(num_classes, emb_size, s, m)
            else:
                self.output = CombineFace(num_classes, emb_size, s, a, m, b)

    def _make_layer(self, block, layers, channels, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, 2, True, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, prefix=''))
        return layer

    def hybrid_forward(self, F, x, y):
        x = self.features(x)
        fc = self.output(x, y)
        return fc


def resnet50(num_classes, **kwargs):
    return ResNet(BasicBlock, layers=[3, 4, 14, 3], channels=[64, 64, 128, 256, 512],
                  num_classes=num_classes, **kwargs)


def resnet100(num_classes, **kwargs):
    return ResNet(BasicBlock, layers=[3, 13, 30, 3], channels=[64, 64, 128, 256, 512],
                  num_classes=num_classes, **kwargs)
