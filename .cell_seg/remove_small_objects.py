import spex_segment as sp


def run(**kwargs):

    new_label = kwargs.get('new_label')
    minsize = kwargs.get('minsize')

    cellpose_label = sp.remove_small_objects(new_label, minsize)

    return {'new_label': cellpose_label}
