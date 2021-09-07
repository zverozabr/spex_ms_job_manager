import re
from os import getenv, path
import pathlib


def get_filename_from_cd(cd):
    if not cd:
        return None
    filename = re.findall('filename=(.+)', cd)
    if len(filename) == 0:
        return None

    return filename[0]


def getAbsoluteRelative(path_, absolute=True):
    if absolute:
        return path_.replace('%DATA_STORAGE%', getenv('DATA_STORAGE'))
    else:
        return path_.replace(getenv('DATA_STORAGE'), '%DATA_STORAGE%')


def download_file(path_, method='get', client=None, jobid='', taskid=''):
    if client is None:
        raise Exception('client is required')

    dir = f'{getenv("DATA_STORAGE")}/{str(jobid)}/{str(taskid)}'
    relative_dir = f'%DATA_STORAGE%/{str(jobid)}/{str(taskid)}'
    with client.get(path_, stream=True) as reader:
        filename = get_filename_from_cd(reader.headers.get('content-disposition'))
        if reader.ok is False:
            return None

        reader.raise_for_status()

        if filename is None:
            return None

        if not path.exists(dir):
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        with open(f'{dir}/{filename}', 'wb') as result:
            for chunk in reader.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                result.write(chunk)
            result.close()

    return f'{relative_dir}/{filename}'
