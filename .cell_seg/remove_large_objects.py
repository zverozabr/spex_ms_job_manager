import spex_segment as sp


def run(**kwargs):

    new_label = kwargs.get('new_label')
    maxsize = kwargs.get('maxsize')

    cellpose_label = sp.remove_large_objects(new_label, maxsize)

    return {'new_label': cellpose_label}
