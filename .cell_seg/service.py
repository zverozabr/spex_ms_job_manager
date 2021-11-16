import pickle
from os import path as pt


def get(name):

    folder = pt.basename(pt.dirname(name))
    filename = f'./{folder}/{pt.basename(name).split(".")[0]}.pickle'
    with open(filename, "rb") as infile:
        data = pickle.load(infile)
        infile.close()

        return data


def put(name, data):

    folder = pt.basename(pt.dirname(name))
    filename = f'./{folder}/{pt.basename(name).split(".")[0]}.pickle'
    outfile = open(filename, "wb")
    pickle.dump(data, outfile)
    outfile.close()
