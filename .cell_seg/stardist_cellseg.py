import spex_segment as sp


def run(**kwargs):

    channel_list = kwargs.get('channel_list')
    median_image = kwargs.get('median_image')
    scaling = kwargs.get('scaling')
    threshold = kwargs.get('threshold')
    _min = kwargs.get('_min')
    _max = kwargs.get('_max')

    stardist_label = sp.stardist_cellseg(median_image, channel_list, scaling, threshold, _min, _max)

    return {'new_label': stardist_label}
