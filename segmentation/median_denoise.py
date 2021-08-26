import spex_segment as sp
import skimage.segmentation as segmentation
import numpy as np
from tifffile import TiffWriter


def run(**kwargs):

    image = kwargs.get('image')
    channel_list = kwargs.get('channel_list')

    # image, channel = sp.load_tiff(image, is_mibi=True)
    median_image = sp.median_denoise(image, 5, channel_list)

    return median_image
