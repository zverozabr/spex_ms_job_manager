# import spex_segment as sp
from service import get, put
from skimage.segmentation import expand_labels


def simulate_cell(_label, dist):

    """Dilate labels by fixed amount to simulate cells

    Parameters
    ----------
    _label: numpy array of segmentation labels
    dist: number of pixels to dilate

    Returns
    -------
    out : 2D label numpy array with simulated cells

    """

    expanded = expand_labels(_label, dist)

    return expanded


def run(**kwargs):

    dist = kwargs.get('dist')
    new_label = kwargs.get('new_label')
    expanded = simulate_cell(new_label, dist)

    return {'new_label': expanded}


if __name__ == "__main__":
    put(__file__, run(**get(__file__)))
