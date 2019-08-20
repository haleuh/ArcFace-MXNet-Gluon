# https://github.com/YadiraF/PRNet/blob/master/predictor.py
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope

import os

import numpy as np
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from skimage.transform import rescale
import tensorflow as tf


def resBlock(x, num_outputs, kernel_size=4, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
             scope=None):
    assert num_outputs % 2 == 0  # num_outputs must be divided by channel_factor(2 here)
    with tf.variable_scope(scope, 'resBlock'):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_outputs:
            shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride,
                                  activation_fn=None, normalizer_fn=None, scope='shortcut')
        x = tcl.conv2d(x, num_outputs / 2, kernel_size=1, stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs / 2, kernel_size=kernel_size, stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)

        x += shortcut
        x = normalizer_fn(x)
        x = activation_fn(x)
    return x


class resfcn256(object):
    def __init__(self, resolution_inp=256, resolution_op=256, channel=3, name='resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, x, is_training=True):
        with tf.variable_scope(self.name):
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
                               normalizer_fn=tcl.batch_norm,
                               biases_initializer=None,
                               padding='SAME',
                               weights_regularizer=tcl.l2_regularizer(0.0002)):
                    size = 16
                    # x: s x s x 3
                    se = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1)  # 256 x 256 x 16
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2)  # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1)  # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2)  # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1)  # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2)  # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1)  # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2)  # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1)  # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=2)  # 8 x 8 x 512
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=1)  # 8 x 8 x 512

                    pd = tcl.conv2d_transpose(se, size * 32, 4, stride=1)  # 8 x 8 x 512
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=2)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=2)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=2)  # 64 x 64 x 64
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64

                    pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=2)  # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=1)  # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(pd, size, 4, stride=2)  # 256 x 256 x 16
                    pd = tcl.conv2d_transpose(pd, size, 4, stride=1)  # 256 x 256 x 16

                    pd = tcl.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
                    pd = tcl.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
                    # padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
                    pos = tcl.conv2d_transpose(pd, 3, 4, stride=1,
                                               activation_fn=tf.nn.sigmoid)

                    return pos

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class PosPrediction:
    def __init__(self, x, y, z, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1

        # network type
        self.network = resfcn256(self.resolution_inp, self.resolution_op)

        # net forward
        self.x = x
        self.y = y
        self.z = z
        self.x_op = self.network(self.x, is_training=False)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    def restore(self, model_path):
        tf.train.Saver(self.network.vars).restore(self.sess, model_path)

    def predict(self):
        x, y, z = self.sess.run([self.x_op, self.y, self.z])
        return x * self.MaxPos, y, z


class PRN:
    """
    Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    """

    def __init__(self, data_dir, data_list, batch_size=64):
        self.data_dir = data_dir
        # resolution of input and output image size.
        self.resolution_inp = 256
        self.resolution_op = 256

        # create dataset
        self.dataset = self.create_dataset(data_dir, data_list, batch_size)
        self.iterator = self.dataset.make_one_shot_iterator()
        self.x, self.y, self.z = self.iterator.get_next()

        # ---- load PRN
        self.pos_predictor = PosPrediction(self.x, self.y, self.z, self.resolution_inp, self.resolution_op)
        self.pos_predictor.restore('shared/PRNet/net-data/256_256_resfcn256_weight')

        # uv file
        self.uv_kpt_ind = np.loadtxt('shared/PRNet/uv-data/uv_kpt_ind.txt').astype(np.int32)  # 2 x 68 get kpt
        # get valid vertices in the pos map
        self.face_ind = np.loadtxt('shared/PRNet/uv-data/face_ind.txt').astype(np.int32)
        self.triangles = np.loadtxt('shared/PRNet/uv-data/triangles.txt').astype(np.int32)  # ntri x 3

        self.uv_coords = self.generate_uv_coords()

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution), range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1, 2, 0])
        uv_coords = np.reshape(uv_coords, [resolution ** 2, -1])
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:, :2], np.zeros([uv_coords.shape[0], 1])))
        return uv_coords

    def net_forward(self):
        return self.pos_predictor.predict()

    def process_next(self):
        """
        process image with crop operation.
        """
        # run our net
        cropped_poss, tform_params, img_paths = self.net_forward()
        kpts = []
        # poses = []
        # restore
        for cropped_pos, tform_param, img_path in zip(cropped_poss, tform_params, img_paths):
            cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
            z = cropped_vertices[2, :].copy() / tform_param[0, 0]
            cropped_vertices[2, :] = 1
            vertices = np.dot(np.linalg.inv(tform_param), cropped_vertices)
            vertices = np.vstack((vertices[:2, :], z))
            pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])
            # save landmarks
            kpt = self.get_landmarks(pos)
            # save pose
            # vertices = self.get_vertices(pos)
            # _, pose = estimate_pose(vertices)
            # pose = np.array(pose) * 180 / np.pi
            kpts.append(kpt[:, :2].flatten())
            # poses.append(pose)
        return kpts

    def get_landmarks(self, pos):
        """
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        """
        kpt = pos[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
        return kpt

    def get_vertices(self, pos):
        """
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        """
        all_vertices = np.reshape(pos, [self.resolution_op ** 2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices

    def set_shape(self, x, y, z):
        x.set_shape([self.resolution_inp, self.resolution_op, 3])
        y.set_shape([3, 3])
        return x, y, z

    def create_dataset(self, data_dir, data_list, batch_size=64):
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(lambda img_path, bbox: tuple(tf.py_func(self.read_image,
                                                                      [data_dir, img_path, bbox],
                                                                      [tf.float32, tf.float32, tf.string])),
                              num_parallel_calls=4)
        dataset = dataset.map(self.set_shape)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)  # prefetch 1 batch, not 1 file
        return dataset

    def read_image(self, data_dir, img_path, bbox):
        full_img_path = os.path.join(data_dir, img_path).decode("utf-8")
        image = imread(full_img_path)
        max_size = max(image.shape[0], image.shape[1])
        if max_size > 1000:
            image = rescale(image, 1000. / max_size)
            image = (image * 255).astype(np.uint8)
        if image.ndim < 3:
            image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        # Read face bbox
        left, top, right, bottom = bbox
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
        size = int(old_size * 1.58)
        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_op - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_op))

        return cropped_image.astype(np.float32), tform.params.astype(np.float32), img_path
