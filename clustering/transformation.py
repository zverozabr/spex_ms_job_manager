from service import get, put
import numpy as np


def run(**kwargs):

    df = kwargs.get('df')

    markers = kwargs.get('markers')

    data = df
    data_for_calc = data[:, markers]
    data_for_calc = np.arcsinh(data_for_calc / 5)
    data[:, markers] = data_for_calc
    return {
        'transformed': data
    }


if __name__ == "__main__":
    put(__file__, run(**get(__file__)))
