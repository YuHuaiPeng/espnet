"""Modified Spec Augment module for preprocessing i.e., data augmentation"""

import random

import numpy
from PIL import Image
from PIL.Image import BICUBIC, NEAREST

from espnet.transform.functional import FuncTrans


def time_stretch(
    x,
    min_segment_length=19,
    max_segment_length=32,
    min_stretch_ratio=0.5,
    max_stretch_ratio=1.5,
    interp_mode="bicubic",
):
    """time stretching for data augmentation

    :param numpy.ndarray x: features (time, freq)
    :param int min_segment_length: minimum segment length. Must be larger than 0.
    :param int max_segment_length: maximum segment length. If not larger than 0, stretch whole feature.
    :param int min_stretch_ratio: minimum ratio
    :param int max_stretch_ratio: maximum ratio
    :param str interp_mode: "bicubic" or "nearest"
    :returns numpy.ndarray: time stretched features (new_time, freq)
    """

    # check interpolation mode
    if type(x) is list:
        assert interp_mode == "nearest"
    if interp_mode == "bicubic":
        interp_mode = BICUBIC
    elif interp_mode == "nearest":
        interp_mode = NEAREST
    else:
        raise NotImplementedError(
            "unknown interpolation mode: "
            + interp_mode
            + ", choose one from (bicubic, nearest)."
        )

    # check segment_length
    if min_segment_length > max_segment_length:
        raise ValueError(
            "min_segment_length must not be larger than max_segment_length."
        )
    if max_segment_length <= 0:
        min_segment_length = x.shape[0]
        max_segment_length = x.shape[0]
    if min_segment_length < 1:
        raise ValueError("min_segment_length must be larger than 0.")

    t = 0
    new_features = []
    
#    if tpe(x) is list:
#        total_time = len(x)
#    else:
    total_time = x.shape[0]
    x = x.reshape(-1, 1)

    while t < total_time:
        segment_length = random.randint(min_segment_length, max_segment_length)
        segment_ratio = random.uniform(min_stretch_ratio, max_stretch_ratio)
        new_segment_length = int(segment_length * segment_ratio)
#        print(x[t:t+segment_length])

        if type(x) is list:
            # input looks like: [[idx, idx], [idx idx], ...]
            segment_image = numpy.array(x[t : t + segment_length]).astype(numpy.uint8)
            stretched_image = Image.fromarray(segment_image).resize(
                (segment_image.shape[1], new_segment_length), interp_mode
            )
            new_segment = numpy.array(stretched_image)
        else:
            new_segment = numpy.array(
                Image.fromarray(x[t : t + segment_length].astype(numpy.uint8)).resize(
                    (x.shape[1], new_segment_length), interp_mode
                )
            )
        new_features.append(new_segment)
        t = t + segment_length
#    print(new_features[0].shape, new_features[1].shape, len(new_features))
#    print(new_features)
    new_features = numpy.concatenate(new_features, axis=0)
    if type(x) is list:
        new_features = new_features.tolist()
#    print(new_features.shape)
    new_features = new_features.reshape(-1)
    return new_features

def utt_stretch(
    x,
    min_stretch_ratio=0.5,
    max_stretch_ratio=1.5,
    interp_mode="bicubic",
):
    """time stretching for data augmentation

    :param numpy.ndarray x: features (time, freq)
    :param int min_segment_length: minimum segment length. Must be larger than 0.
    :param int max_segment_length: maximum segment length. If not larger than 0, stretch whole feature.
    :param int min_stretch_ratio: minimum ratio
    :param int max_stretch_ratio: maximum ratio
    :param str interp_mode: "bicubic" or "nearest"
    :returns numpy.ndarray: time stretched features (new_time, freq)
    """

    # check interpolation mode
    if type(x) is list:
        assert interp_mode == "nearest"
    if interp_mode == "bicubic":
        interp_mode = BICUBIC
    elif interp_mode == "nearest":
        interp_mode = NEAREST
    else:
        raise NotImplementedError(
            "unknown interpolation mode: "
            + interp_mode
            + ", choose one from (bicubic, nearest)."
        )

    t = 0
    new_features = []
    
#    if tpe(x) is list:
#        total_time = len(x)
#    else:
    total_time = x.shape[0]
    x = x.reshape(-1, 1)

    segment_length = x.shape[0]
    segment_ratio = random.uniform(min_stretch_ratio, max_stretch_ratio)
    new_segment_length = int(segment_length * segment_ratio)
#        print(x[t:t+segment_length])

    if type(x) is list:
        # input looks like: [[idx, idx], [idx idx], ...]
        segment_image = numpy.array(x).astype(numpy.uint8)
        stretched_image = Image.fromarray(segment_image).resize(
            (segment_image.shape[1], new_segment_length), interp_mode
        )
        new_segment = numpy.array(stretched_image)
    else:
        new_segment = numpy.array(
            Image.fromarray(x.astype(numpy.uint8)).resize(
                (x.shape[1], new_segment_length), interp_mode
            )
        )
        new_features.append(new_segment)
#    print(new_features[0].shape, new_features[1].shape, len(new_features))
#    print(new_features)
    new_features = numpy.concatenate(new_features, axis=0)
    if type(x) is list:
        new_features = new_features.tolist()
#    print(new_features.shape)
    new_features = new_features.reshape(-1)

    return new_features
def utt_seg_stretch(
    x,
    min_segment_length=19,
    max_segment_length=32,
    min_stretch_ratio=0.5,
    max_stretch_ratio=1.5,
    interp_mode="bicubic",
):
    return time_stretch(
              utt_stretch(x,
                          min_stretch_ratio,
                          max_stretch_ratio,
                          interp_mode),
              min_segment_length, 
              max_segment_length, 
              min_stretch_ratio, 
              max_stretch_ratio, 
              interp_mode)
class UttSegStretch(FuncTrans):
    _func = utt_seg_stretch
    __doc__ = time_stretch.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)
class UttStretch(FuncTrans):
    _func = utt_stretch
    __doc__ = time_stretch.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)
class TimeStretch(FuncTrans):
    _func = time_stretch
    __doc__ = time_stretch.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)
