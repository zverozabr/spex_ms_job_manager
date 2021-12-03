import numpy as np
from skimage.filters import median, gaussian
from skimage.morphology import watershed, dilation, erosion, disk, binary_dilation
from skimage.util import apply_parallel
from service import get, put

def median_denoise(image, kernal, ch):

    """Non local means denoising

    Parameters
    ----------
    image : Multichannel numpy array (C,X,Y)
    kernal: int, 5-7 is a typical range
    ch: list of int, indexes of channels to be denoised

    Returns
    -------
    Image Stack : Denoised image stack as numpy array (C,X,Y)

    """

    global filtered
    filter_channels = ch

    def median_denoise_wrap(array):
        correct = array[0]
        correct = median(correct, disk(kernal))
        return correct[np.newaxis, ...]

    denoise = apply_parallel(
        median_denoise_wrap,
        image,
        chunks=(1, image.shape[1], image.shape[2]),
        dtype="float",
        compute=True,
    )

    for i in range(0, image.shape[0], 1):
        if i in filter_channels:
            temp = denoise[i]
            temp = np.expand_dims(temp, 0)
            if i == 0:
                filtered = temp
            else:
                filtered = np.concatenate((temp, filtered), axis=0)
        else:
            temp = image[i]
            temp = np.expand_dims(temp, 0)
            if i == 0:
                filtered = temp
            else:
                filtered = np.concatenate((temp, filtered), axis=0)

    f_denoise = np.flip(filtered, axis=0)

    return f_denoise


def run(**kwargs):

    image = kwargs.get('median_image')
    channel_list = kwargs.get('channel_list', [0])
    kernal = kwargs.get('kernal', 5)
    if isinstance(kernal, str):
        kernal = int(kernal)
    median_image = median_denoise(image, kernal, channel_list)

    return {'median_image': median_image}


if __name__ == '__main__':
    put(__file__, run(**get(__file__)))
