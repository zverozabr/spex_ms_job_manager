import spex_segment as sp


def run(**kwargs):

    channel_list = kwargs.get('channel_list')
    image = kwargs.get('image')
    new_label = None
    if kwargs.get('stardist_label') is not None:
        new_label = sp.rescue_cells(image, channel_list, kwargs.get('stardist_label'))
    if kwargs.get('deepcell_label') is not None:
        new_label = sp.rescue_cells(image, channel_list, kwargs.get('deepcell_label'))
    if kwargs.get('cellpose_label') is not None:
        new_label = sp.rescue_cells(image, channel_list, kwargs.get('cellpose_label'))

    return {'new_label': new_label}
