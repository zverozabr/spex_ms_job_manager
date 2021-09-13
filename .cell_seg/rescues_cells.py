import spex_segment as sp


def run(**kwargs):

    channel_list = kwargs.get('channel_list')
    image = kwargs.get('image')
    new_label = kwargs.get('new_label')

    new_label = sp.rescue_cells(image, channel_list, new_label)

    return {'new_label': new_label}
