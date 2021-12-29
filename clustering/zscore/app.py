import scipy.stats as stats
import numpy as np


def run(**kwargs):

    fn_in = kwargs.get('transformed')
    # 5,7,8,9,11,12,15,16,17,18,19,21,22,24,26,27

    markers = kwargs.get('markers')

    # with open(fn_in, 'r') as f:
    #     reader = csv.reader(f, delimiter=',')
    #     headers = next(reader)
    #     data = np.array(list(reader)).astype(float)
    data = fn_in
    data_for_calc = data[:, markers]
    data_for_calc = stats.zscore(data_for_calc, axis=0)
    data_for_calc = np.nan_to_num(data_for_calc)
    data[:, markers] = data_for_calc
    return {
        'z_score': data,
        'markers': markers
    }
