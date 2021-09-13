import spex_segment as sp


def run(**kwargs):

    image = kwargs.get('image_path')
    image, channel = sp.load_tiff(image, is_mibi=True)

    return {
        'median_image': image,
        'channel': channel,
        'image': image
    }
