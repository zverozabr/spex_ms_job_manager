from app import run
import pickle
import os


def get(name):
    filename, ext = os.path.splitext(name)
    filename = f'{filename}.pickle'

    with open(filename, "rb") as infile:
        data = pickle.load(infile)
        return data


def put(name, data):
    filename, ext = os.path.splitext(name)
    filename = f'{filename}.pickle'

    with open(filename, "wb") as outfile:
        pickle.dump(data, outfile)


if __name__ == '__main__':
    print(f'current env: {os.environ["VIRTUAL_ENV"]}')
    result = run(**get(__file__))
    put(__file__, result)
