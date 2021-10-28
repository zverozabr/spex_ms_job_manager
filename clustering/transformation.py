import glob
import os
import time
import sys

import csv
import numpy as np


def run(**kwargs):

    fn_in = kwargs.get('fn_in')

    markers = kwargs.get('markers')

    data = fn_in
    data_for_calc = data[:, markers]
    data_for_calc = np.arcsinh(data_for_calc / 5)
    data[:, markers] = data_for_calc
    return {
        'transformed': data
    }
