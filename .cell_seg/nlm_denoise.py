import spex_segment as sp


def run(**kwargs):

    image = kwargs.get('median_image')
    patch = kwargs.get('patch') or 5
    dist = kwargs.get('dist') or 6
    median_image = sp.nlm_denoise(image, patch, dist)

    return {'median_image': median_image}
