import spex_segment as sp


def run(**kwargs):

    channel_list = kwargs.get('channel_list')
    median_image = kwargs.get('median_image')

    dilated_labels = sp.classicwatershed_cellseg(median_image, channel_list)

    return {'new_label': dilated_labels}
