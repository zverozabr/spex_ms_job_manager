import spex_segment as sp


def run(**kwargs):

    dist = kwargs.get('dist')
    new_label = kwargs.get('new_label')
    expanded = sp.simulate_cell(new_label, dist)

    return {'new_label': expanded}
