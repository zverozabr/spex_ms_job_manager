# import spex_segment as sp
import numpy as np
from skimage.measure import label, regionprops_table, regionprops
from skimage.filters import median, gaussian
from skimage.feature import peak_local_max
from skimage.morphology import watershed, dilation, erosion, disk, binary_dilation
import math
from service import get, put


def rescue_cells(image, seg_channels, label_ling):

    """Rescue/Segment cells that deep learning approach may have missed

    Parameters
    ----------
    image : raw image 2d numpy array
    seg_channels: list of indices to use for nuclear segmentation
    label_ling: numpy array of segmentation labels

    Returns
    -------
    combine_label : 2D numpy array with added cells

    """
    temp2 = np.zeros((image.shape[1], image.shape[2]))
    for i in seg_channels:
        try:
            temp = image[i]
            temp2 = temp + temp2
        except IndexError:
            print("oops")

    seg_image = temp2 / len(seg_channels)

    props = regionprops_table(
        label_ling, intensity_image=seg_image, properties=["mean_intensity", "area"]
    )

    meanint_cell = np.mean(props["mean_intensity"])
    meansize_cell = np.mean(props["area"])
    if np.isnan(meansize_cell):
        print("oops")
        meansize_cell = 0
    if np.isnan(meanint_cell):
        print("oops")
        meanint_cell = 0
    radius = math.floor(math.sqrt(meansize_cell / 3.14) * 0.5)
    threshold = meanint_cell * 0.5

    med = median(seg_image, disk(radius))
    local_max = peak_local_max(
        med, min_distance=math.floor(radius * 1.2), indices=False
    )

    mask = med > threshold

    mask = binary_dilation(mask, np.ones((2, 2)))
    masked_peaks = local_max * mask

    seed_label = label(masked_peaks)

    watershed_labels = watershed(
        image=-med, markers=seed_label, mask=mask, watershed_line=True, compactness=20
    )

    selem = disk(1)
    dilated_labels = erosion(watershed_labels, selem)
    selem = disk(1)
    dilated_labels = dilation(dilated_labels, selem)

    labels2 = label_ling > 0

    props = regionprops(dilated_labels, intensity_image=labels2)

    labels_store = np.arange(np.max(dilated_labels) + 1)

    for cell in props:
        if cell.mean_intensity >= 0.03:
            labels_store[cell.label] = 0

    final_mask = labels_store[dilated_labels]

    combine_label = label_ling + final_mask
    combine_label = label(combine_label)

    return combine_label


def run(**kwargs):

    channel_list = kwargs.get('channel_list')
    image = kwargs.get('image')
    new_label = kwargs.get('new_label')

    new_label = rescue_cells(image, channel_list, new_label)

    return {'new_label': new_label}


if __name__ == "__main__":
    put(__file__, run(**get(__file__)))
