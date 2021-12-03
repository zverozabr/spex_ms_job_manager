# import spex_segment as sp
from service import get, put
import numpy as np


def remove_large_objects(segments, maxsize):

    """Remove large segmented objects

    Parameters
    ----------
    segments: numpy array of segmentation labels
    maxsize: max pixel size

    Returns
    -------
    out : 2D label numpy array

    """

    out = np.copy(segments)
    component_sizes = np.bincount(segments.ravel())

    too_large = component_sizes > maxsize
    too_large_mask = too_large[segments]
    out[too_large_mask] = 0

    return out


def run(**kwargs):

    new_label = kwargs.get('new_label')
    maxsize = int(kwargs.get('maxsize'))

    cellpose_label = remove_large_objects(new_label, maxsize)

    return {'new_label': cellpose_label}


if __name__ == "__main__":
    put(__file__, run(**get(__file__)))
