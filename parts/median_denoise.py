import spex_segment as sp
import skimage.segmentation as segmentation
import numpy as np
from tifffile import TiffWriter


def run(image='', channel_list=[]):

    image = '2.ome.tiff'
    channel_list = [0, 1, 3]

    image, channel = sp.load_tiff(image, is_mibi=True)
    median_image = sp.median_denoise(image, 5, channel_list)

    return median_image
