import spex_segment as sp


def run(**kwargs):

    image = kwargs.get('median_image')
    ch = kwargs.get('ch')
    top = kwargs.get('top')
    subtraction = kwargs.get('subtraction')

    median_image = sp.background_subtract(image, ch, top, subtraction)

    return {'median_image': median_image}
