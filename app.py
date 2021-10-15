from spex_common.config import load_config
from spex_common.modules.database import db_instance
from spex_common.models.Task import task
from spex_common.models.Job import job
import spex_common.modules.omeroweb as omeroweb
import pathlib
from os import path
from spex_common.modules.logging import get_logger
from spex_common.modules.aioredis import send_event
from spex_common.models.OmeroImageFileManager import (
    OmeroImageFileManager as FileManager,
)
from glob import glob
import json
import importlib
import os
from services.Utils import getAbsoluteRelative
from services.Timer import every
import sys
import pickle

logger = get_logger(f"spex.ms-job-manager")

collection = "tasks"


def scripts_list():
    scripts: list = []
    folder = f'{os.getenv("DATA_STORAGE")}\\Scripts\\*\\'
    for script_folder in glob(folder):
        scripts += [os.path.basename(os.path.dirname(script_folder))]
    return scripts


def get_script_params(script: str = "", part: str = "", subpart: list = None):
    subpart = subpart if subpart else []
    _result: list = []
    for file in glob(f"{script}/{part}.json", recursive=True):
        data = json.load(open(file))

        if depends := data.get("depends_and_script"):
            for item in depends:
                _result += get_script_params(script=script, part=item, subpart=subpart)
        if depends := data.get("depends_or_script"):
            if scripts := set(subpart).intersection(set(depends)):
                for item in scripts:
                    _result += get_script_params(
                        script=script, part=item, subpart=subpart
                    )
        logger.info(f"{script}-{part}")

        if params := data.get("start_params"):
            _result += [{part: params}]

    return _result


def start_scenario(
    script: str = "",
    part: str = "",
    subpart: list = None,
    folder: str = "",
    and_scripts: list = None,
    start_depends: bool = True,
    **kwargs,
):
    subpart = subpart if subpart else []
    and_scripts = and_scripts if and_scripts else []
    for file in glob(f"{folder}/{part}.json", recursive=True):
        data = json.load(open(file))

        if start_depends is True:
            if depends := data.get("depends_and_script"):
                for item in depends:
                    res = start_scenario(
                        script=script,
                        part=item,
                        subpart=subpart,
                        folder=folder,
                        and_scripts=and_scripts,
                        **kwargs,
                    )
                    kwargs.update(res)
            if depends := data.get("depends_or_script"):
                if scripts := set(subpart).intersection(set(depends)):
                    for item in scripts:
                        res = start_scenario(
                            script=script,
                            part=item,
                            subpart=subpart,
                            folder=folder,
                            and_scripts=and_scripts,
                            **kwargs,
                        )
                        kwargs.update(res)

        logger.info(f"{script}-{part}")
        params = data.get("start_params")
        for item in params:
            allowed_keys = ["or", "and"]
            key_name = list(item.keys())[0]
            if kwargs.get(key_name) is None and key_name not in allowed_keys:
                raise ValueError(
                    f"Not have param {key_name} in script: {script}, in part {part}"
                )
            elif key_name == "or":
                have_data = False
                for param in item.get(key_name):
                    sub_item = list(param.keys())[0]
                    if kwargs.get(sub_item) is not None:
                        have_data = True
                if not have_data:
                    raise ValueError(
                        f"Not have any of: {item.get(key_name)} params in script: {script}, in part {part}"
                    )
            elif key_name == "and":
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
        sys.path.append(folder)
        module = importlib.import_module(data["script_path"])
        res = module.run(**kwargs)

        kwargs.update(res)

        if and_scripts:
            stages = get_script_structure(script)
            for _script in and_scripts:
                cur_stage = data.get("stage")
                if item := [
                    element
                    for element in stages.get(cur_stage)
                    if element["script_path"] == _script
                ]:
                    res = start_scenario(
                        script=script,
                        part=_script,
                        subpart=subpart,
                        folder=folder,
                        start_depends=False,
                        **kwargs,
                    )
                    kwargs.update(res)

        return kwargs


def get_task():
    tasks = db_instance().select(
        collection,
        "FILTER doc.status == 0 or doc.status == -1 and doc.content like @value "
        "LIMIT 1 ",
        value="%empty%",
    )
    if len(tasks) == 1:
        return task(tasks[0]).to_json()
    else:
        tasks = db_instance().select(collection, " FILTER doc.status == 0 LIMIT 1  ")
        return task(tasks[0]).to_json() if len(tasks) == 1 else None


def get_script_structure(folder: str = None):
    result_data = {}
    folder = f'{os.getenv("DATA_STORAGE")}/Scripts/{folder}/'
    for file in glob(f"{folder}stages.json", recursive=True):
        data = json.load(open(file))
        for key, value in data.items():
            for file_scr in glob(f"{folder}*.json", recursive=True):
                file_scr_data = json.load(open(file_scr))
                if file_scr_data.get("stage") == key:
                    cur = result_data.get(key) if result_data.get(key) else []
                    result_data.update({key: cur + [file_scr_data]})
        result_data.update(stages=data)
    return result_data


def get_image_from_omero(a_task):
    image_id = a_task["omeroId"]
    _path = None
    file = FileManager(image_id)
    if not file.exists():

        author = a_task.get("author").get("login")

        send_event(
            "omero/download/image", {"id": image_id, "override": False, "user": author}
        )
    else:
        _path = file.get_filename()

    return _path


def get_path(_jobid, _taskid):
    _dir = f'{os.getenv("DATA_STORAGE")}/{str(_jobid)}/{str(_taskid)}'
    if not path.exists(_dir):
        pathlib.Path(_dir).mkdir(parents=True, exist_ok=True)

    return f"{_dir}"


def update_status(status, onetask, result=None):
    search = "FILTER doc._key == @value LIMIT 1"
    data = {"status": status}
    if result:
        data.update({"result": result})
    db_instance().update(collection, data, search, value=onetask["id"])


def enrich_task_data(a_task):

    parent_jobs = db_instance().select(
        'pipeline_direction',
        "FILTER doc._to == @value",
        value=f"jobs/{a_task['parent']}",
    )
    if parent_jobs:
        jobs_ids = [item['_from'].replace('jobs/', '') for item in parent_jobs]
        tasks = db_instance().select(
            'tasks',
            'FILTER doc.parent in @value '
            'and doc.result != "" '
            'and doc.result != Null ',

            value=jobs_ids,
        )
        data = {}
        for _task in tasks:
            filename = getAbsoluteRelative(_task['result'], True)
            with open(filename, "rb") as outfile:
                current_file_data = pickle.load(outfile)
                data = {**data, **current_file_data}

    return data


def take_start_return_result():
    a_task = get_task()
    if a_task is not None:
        a_task["params"] = {**enrich_task_data(a_task), **a_task["params"]}
        # download image tiff
        if not a_task["params"].get("image_path"):
            _path = get_image_from_omero(a_task)
        else:
            _path = a_task["params"].get("image_path")

        if _path is None:
            update_status(-1, a_task)
            return None

        script_path = getAbsoluteRelative(
            f'{os.getenv("DATA_STORAGE")}/Scripts/{a_task["params"]["script"]}'
        )

        filename = f"{get_path(a_task['id'], a_task['parent'])}/result.pickle"
        if os.path.isfile(_path):
            a_task["params"].update(image_path=_path)
            a_task["params"].update(folder=script_path)
            result = start_scenario(**a_task["params"], start_depends=False)
            if not result:
                logger.info(f'problems with scenario params {a_task["params"]}')
            else:
                outfile = open(filename, "wb")
                pickle.dump(result, outfile)
                outfile.close()
        if os.path.isfile(filename):
            update_status(100, a_task, result=getAbsoluteRelative(filename, False))
            logger.info("1 task complete")
    else:
        logger.info("0 task")


if __name__ == "__main__":
    load_config()
every(5, take_start_return_result)
# take_start_return_result()

# result = start_scenario(
#     script="segmentation",
#     part="denoise",
#     image_path="2.ome.tiff",
#     channel_list=[0, 2, 3],
# )

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


# result = start_scenario(
#     script='cell_seg',
#     part='nlm_denoise',
#     and_scripts=['median_denoise', 'background_subtract'],
#     # image_path='2.ome.tiff',
#     folder=".cell_seg",
#     channel_list=[0, 2, 3],
#     threshold=0.5,
#     _min=1,
#     _max=98.5,
#     scaling=1,
#     image_id=1,
#     kernal=5,
#     mpp=0.39,
#     ch=0,
#     top=5,
#     subtraction=10,
#     dist=8)

#
# result = start_scenario(
#     script='cell_seg',
#     part='feature_extraction',
#     subpart=[
#         'stardist_cellseg',
#         'median_denoise'
#     ],
#     and_scripts=[
#         'remove_small_objects',
#         'background_subtract',
#         'remove_large_objects'
#     ],
#     folder='.cell_seg',
#     image_path='2.ome.tiff',
#     channel_list=[0, 2, 3],
#     scaling=1,
#     kernal=5,
#     _min=1,
#     _max=98.5,
#     threshold=0.5,
#     mpp=0.39,
#     ch=0,
#     top=20,
#     subtraction=1,
#     minsize=2,
#     maxsize=98,
#     dist=8)
#
# print(result)

