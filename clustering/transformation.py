import glob
import os
import time
import sys

import csv
import numpy as np


def run(**kwargs):

    fn_in = kwargs.get('fn_in')
    # fn_in = sys.argv[1]
    fn_out = 'test'

    # 5,7,8,9,11,12,15,16,17,18,19,21,22,24,26,27
    # marker_list = sys.argv[3].split(',')
    markers = kwargs.get('markers')
    # markers = [int(x) for x in marker_list]

    with open(fn_in, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = np.array(list(reader)).astype(float)
    data_for_calc = data[:, markers]
    data_for_calc = np.arcsinh(data_for_calc / 5)
    data[:, markers] = data_for_calc
    # np.savetxt(
    #     fn_out,
    #     data,
    #     fmt='%s',
    #     delimiter=',', header=','.join(headers), comments=''
    # )
    return {
      'transformed': data
    }
