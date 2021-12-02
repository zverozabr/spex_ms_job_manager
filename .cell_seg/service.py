import pickle
from os import path as pt
from sys import platform


def get(name):
    if platform != 'linux' and platform != 'linux2':
        # folder = pt.basename(pt.dirname(name))
        # filename = f'{folder}/{pt.basename(name).split(".")[0]}.pickle'
        filename = str(name).replace('.py', '.pickle')
    else:
        filename = f'{name.split(".")[0]}.pickle'
    with open(filename, "rb") as infile:
        data = pickle.load(infile)
        infile.close()

        return data


def put(name, data):
    if platform != 'linux' and platform != 'linux2':
        folder = pt.basename(pt.dirname(name))
        # filename = f'{folder}/{pt.basename(name).split(".")[0]}.pickle'
        filename = str(name).replace('.py', '.pickle')
    else:
        filename = f'{name.split(".")[0]}.pickle'

    outfile = open(filename, "wb")
    pickle.dump(data, outfile)
    outfile.close()
