from spex_common.config import load_config
from spex_common.modules.database import db_instance
from spex_common.models.Task import task
import spex_common.modules.omeroweb as omeroweb
from glob import glob
import json
import importlib.util
import importlib
import os
from services.Utils import download_file, getAbsoluteRelative

collection = 'tasks'


def scripts_list():
    scripts: list = []
    folder = f'{os.getenv("DATA_STORAGE")}\\Scripts\\*\\'
    for script_folder in glob(folder):
        scripts += [os.path.basename(os.path.dirname(script_folder))]
    return scripts


def get_script_params(script: str = "", part: str = "", subpart: list = None):
    subpart = subpart if subpart else []
    result: list = []
    for file in glob(f'{script}/{part}.json', recursive=True):
        data = json.load(open(file))

        if depends := data.get('depends_and_script'):
            for item in depends:
                result += get_script_params(script=script, part=item, subpart=subpart)
        if depends := data.get('depends_or_script'):
            if scripts := set(subpart).intersection(set(depends)):
                for item in scripts:
                    result += get_script_params(script=script, part=item, subpart=subpart)

        print(script, part)
        if params := data.get('start_params'):
            result += [{part: params}]

    return result


def start_scenario(script: str = "", part: str = "", subpart: list = None, **kwargs):
    subpart = subpart if subpart else []
    for file in glob(f'{script}/{part}.json', recursive=True):
        data = json.load(open(file))

        if depends := data.get('depends_and_script'):
            for item in depends:
                res = start_scenario(script=script, part=item, subpart=subpart, **kwargs)
                kwargs.update(res)
        if depends := data.get('depends_or_script'):
            if scripts := set(subpart).intersection(set(depends)):
                for item in scripts:
                    res = start_scenario(script=script, part=item, subpart=subpart, **kwargs)
                    kwargs.update(res)

        module = importlib.import_module(f'.{data["script_path"]}', package=script)
        print(script, part)
        params = data.get('start_params')
        for item in params:
            allowed_keys = ['or', 'and']
            key_name = list(item.keys())[0]
            if kwargs.get(key_name) is None and key_name not in allowed_keys:
                raise ValueError(f"Not have param {key_name} in script: {script}, in part {part}")
            elif key_name == 'or':
                have_data = False
                for param in item.get(key_name):
                    sub_item = list(param.keys())[0]
                    if kwargs.get(sub_item) is not None:
                        have_data = True
                if not have_data:
                    raise ValueError(
                        f"Not have any of: {item.get(key_name)} params in script: {script}, in part {part}"
                    )
            elif key_name == 'and':
                have_all_data = True
                for param in item.get(key_name):
                    sub_item = list(param.keys())[0]
                    if kwargs.get(sub_item) is None:
                        have_all_data = False
                        break
                if not have_all_data:
                    raise ValueError(
                        f"Not have all of: {item.get(key_name)} params in script: {script}, in part {part}"
                    )

        res = module.run(**kwargs)
        kwargs.update(res)
        return kwargs


# result = start_scenario(script='segmentation', part='denoise', image_path='2.ome.tiff', channel_list=[0, 2, 3])
#  1, 0.5, 1, 98.5
# result = start_scenario(
#     script='segmentation',
#     part='stardist_cellseg',
#     image_path='2.ome.tiff',
#     kernal=5,
#     channel_list=[0, 2, 3],
#     scaling=1,
#     threshold=0.5,
#     _min=1,
#     _max=98.5
# )

# result = start_scenario(
#          script='segmentation',
#          part='deepcell_segmentation',
#          image_path='2.ome.tiff',
#          channel_list=[0, 2, 3],
#          kernal=5,
#          mpp=0.39)

# result = start_scenario(
#     script='segmentation',
#     part='rescues_cells',
#     image_path='2.ome.tiff',
#     channel_list=[0, 2, 3],
#     kernal=5,
#     mpp=0.39)


# result = start_scenario(
#     script='segmentation',
#     part='feature_extraction',
#     image_path='2.ome.tiff',
#     channel_list=[0, 2, 3],
#     kernal=5,
#     mpp=0.39,
#     dist=8)


# result = start_scenario(
#     script='segmentation',
#     part='find_boundaries',
#     subpart=['stardist_cellseg'],
#     image_path='2.ome.tiff',
#     channel_list=[0, 2, 3],
#     threshold=0.5,
#     _min=1,
#     _max=98.5,
#     scaling=1,
#     kernal=5,
#     mpp=0.39,
#     dist=8)
#
# print(result)


# params_res = get_script_params(
#     script='segmentation',
#     part='find_boundaries',
#     subpart=['stardist_cellseg']
# )
#
# print(params_res)

def get_task():
    tasks = db_instance().select(collection, 'FILTER doc.status == 0 or doc.status == -1 and doc.content like @value '
                                             'LIMIT 1 ', value='%empty%')
    if len(tasks) == 1:
        return task(tasks[0]).to_json()
    else:
        tasks = db_instance().select(collection, ' FILTER doc.status == 0 LIMIT 1  ')
        return task(tasks[0]).to_json() if len(tasks) == 1 else None


def get_script_structure(folder: str = None):
    result_data = {}
    folder = f'{os.getenv("DATA_STORAGE")}\\Scripts\\{folder}\\'
    for file in glob(f'{folder}stages.json', recursive=True):
        data = json.load(open(file))
        for key, value in data.items():
            for file_scr in glob(f'{folder}*.json', recursive=True):
                file_scr_data = json.load(open(file_scr))
                if file_scr_data.get('stage') == key:
                    cur = result_data.get(key) if result_data.get(key) else []
                    result_data.update({key: cur + [file_scr_data]})
        result_data.update(stages=data)
    return result_data


def get_image_from_omero(a_task):
    image_id = a_task['omeroId']
    author = a_task.get('author').get('login')
    job_id = a_task.get('parent')
    task_id = a_task.get('id')

    session = omeroweb.get(author)

    if session is not None:
        download_url = '/webclient/render_image_download/' + str(image_id) + '/?format=tif'
        path = download_file(download_url, client=session, jobid=job_id, taskid=task_id)
        return path


def update_status(status, onetask):
    search = 'FILTER doc._key == @value LIMIT 1'
    db_instance().update(collection, {'status': status}, search, value=onetask['id'])


def take_start_return_result():
    a_task = get_task()
    if a_task is not None:
        # download image tiff
        path = get_image_from_omero(a_task)

        if path is None:
            update_status(-1, a_task)
            return None
        if os.path.isfile(getAbsoluteRelative(path)):
            a_task['params'].update(image_path=getAbsoluteRelative(path))
            a_task['params']["script"] = f'{os.getenv("DATA_STORAGE")}\\Scripts\\{a_task["params"]["script"]}'
            result = start_scenario(**a_task["params"])
            print(result)


if __name__ == '__main__':
    load_config()
    take_start_return_result()

