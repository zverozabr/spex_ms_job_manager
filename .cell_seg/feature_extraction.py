import spex_segment as sp


def run(**kwargs):

    image = kwargs.get('image')
    new_label = kwargs.get('new_label')
    channel_list = kwargs.get('channel_list')

    df = sp.feature_extraction(image, new_label, channel_list)

    return {'df': df}
