# import spex_segment as sp
from service import get, put
import numpy as np
import cv2
from cellpose import models


def cellpose_cellseg(img, seg_channels, diamtr, scaling):

    """Segment image by cellpose deeplearning method

    Parameters
    ----------
    image : Multichannel image as numpy array
    seg_channels: list of indices to use for nuclear segmentation
    diamtr: typical size of nucleus
    scaling: Integer value scaling

    Returns
    -------
    labels_final : per cell segmentation as numpy array

    """
    temp2 = np.zeros((img.shape[1], img.shape[2]))
    for i in seg_channels:
        temp = img[i]
        temp2 = temp + temp2

    SegImage = temp2
    SegImage = cv2.resize(
        SegImage,
        (SegImage.shape[1] * scaling, SegImage.shape[0] * scaling),
        interpolation=cv2.INTER_NEAREST,
    )

    # model = models.Cellpose(
    #     device=mxnet.cpu(), torch=False, gpu=False, model_type="nuclei"
    # )
    model = models.Cellpose(gpu=False, model_type="nuclei")

    labels, _, _, _ = model.eval(
        [SegImage[::1, ::1]], channels=[[0, 0]], diameter=diamtr
    )

    labels2 = np.float32(labels[0])

    labels_final = cv2.resize(
        labels2,
        (labels2.shape[1] // scaling, labels2.shape[0] // scaling),
        interpolation=cv2.INTER_NEAREST,
    )

    labels_final = np.uint32(labels_final)

    return labels_final


def run(**kwargs):

    median_image = kwargs.get('median_image')
    channel_list = kwargs.get('channel_list')
    scaling = int(kwargs.get('scaling'))
    diamtr = int(kwargs.get('diamtr'))

    cellpose_label = cellpose_cellseg(median_image, channel_list, diamtr, scaling)

    data = {'new_label': cellpose_label}

    put(__file__, data)


if __name__ == '__main__':
    run(**get(__file__))
