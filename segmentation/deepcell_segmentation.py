import spex_segment as sp


def run(**kwargs):

    channel_list = kwargs.get('channel_list')
    median_image = kwargs.get('median_image')
    mpp = kwargs.get('mpp')

    deepcell_label = sp.deepcell_segmentation(median_image, channel_list, mpp)

    return {'deepcell_label': deepcell_label}
