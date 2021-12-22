import numpy as np


def run(**kwargs):

    df = kwargs.get('df')

    markers = kwargs.get('markers')

    data = df.to_numpy()

    try:
        data_for_calc = data[:, markers]
    except:
        markers = [0, 1]
        data_for_calc = data[:, markers]
    data_for_calc = np.arcsinh(data_for_calc / 5)
    data[:, markers] = data_for_calc
    return {
        'transformed': data,
        'markers': markers
    }
