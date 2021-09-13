import spex_segment as sp


def run(**kwargs):

    median_image = kwargs.get('median_image')
    channel_list = kwargs.get('channel_list')
    scaling = kwargs.get('scaling')
    diamtr = kwargs.get('diamtr')

    cellpose_label = sp.cellpose_cellseg(median_image, channel_list, diamtr, scaling)

    return {'new_label': cellpose_label}
