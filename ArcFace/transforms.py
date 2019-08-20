from mxnet.gluon.nn import HybridBlock


class ToTensor(HybridBlock):
    """Converts an image NDArray to a tensor NDArray.

    Converts an image NDArray of shape (H x W x C) in the range
    [0, 255] to a float32 tensor NDArray of shape (C x H x W) in
    the range (-1, 1).

    Inputs:
        - **data**: input tensor with (H x W x C) shape and uint8 type.

    Outputs:
        - **out**: output tensor with (C x H x W) shape and float32 type.
    """
    def __init__(self, **kwargs):
        super(ToTensor, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        x = x.astype('float32')
        x = x.transpose((2, 0, 1))
        x = x - 127.5
        x = x * 0.0078125
        return x
