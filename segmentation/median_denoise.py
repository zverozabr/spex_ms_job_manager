import spex_segment as sp


def run(**kwargs):

    image = kwargs.get('image')
    channel_list = kwargs.get('channel_list')
    kernal = kwargs.get('kernal')
    median_image = sp.median_denoise(image, kernal, channel_list)

    return {'median_image': median_image}
