from os import cpu_count, getenv
from spex_common.models.Task import task
from spex_common.models.History import history
import shutil
import os
import uuid
import json
import pickle
from multiprocessing import Process
import subprocess
from spex_common.modules.logging import get_logger
from spex_common.modules.aioredis import create_aioredis_client
from spex_common.modules.database import db_instance
from spex_common.services.Utils import getAbsoluteRelative
from spex_common.modules.aioredis import send_event
from spex_common.models.OmeroImageFileManager import (
    OmeroImageFileManager as FileManager,
)
import logging
from datetime import datetime


EVENT_TYPE = 'backend/start_job'
MIN_CHUNK_SIZE = 1024 * 1024 * 10
collection = 'tasks'
logger = get_logger()


def add_hist(parent, content):
    db_instance().insert('history', history({
        'author': {'login': 'job_manager_runner', 'id': '0'},
        'date': str(datetime.now()),
        'content': content,
        'parent': parent,
    }).to_json())


def can_start(task_id):
    last_records = db_instance().select(
        'history',
        "FILTER doc.parent == @value SORT doc.date DESC LIMIT 3 ",
        value=task_id,
    )
    key_arr = [record["_key"] for record in last_records]
    if not key_arr:
        return True
    last_canceled_records = db_instance().select(
        'history',
        "FILTER doc.parent == @value"
        " and (doc.content Like @content "
        " or doc.content Like @content2) "
        "SORT doc.date DESC LIMIT 3 ",
        value=task_id,
        content="%-1 to: 1%",
        content2="%1 to: 2%"
    )
    key_arr_2 = [record["_key"] for record in last_canceled_records]
    if key_arr_2 == key_arr and len(key_arr) == 3:
        return False
    return True


def get_platform_venv_params(script, part):
    env_path = os.getenv("SCRIPTS_ENVS_PATH", "~/scripts_envs")

    script_copy_path = os.path.join(env_path, "scripts", script, part)
    os.makedirs(script_copy_path, exist_ok=True)

    env_path = os.path.join(env_path, "envs", script)
    os.makedirs(env_path, exist_ok=True)
    env_path = os.path.join(env_path, part)

    not_posix = os.name != "posix"

    executor = f"python" if not_posix else f"python3"
    start_script = "source" if not_posix else "."
    create_venv = f"{executor} -m venv {env_path}"

    activate_venv = f"{start_script} {os.path.join(env_path, 'bin', 'activate')}"
    if not_posix:
        activate_venv = os.path.join(env_path, "Scripts", "activate.bat")

    return {
        "env_path": env_path,
        "script_copy_path": script_copy_path,
        "create_venv": create_venv,
        "activate_venv": activate_venv,
        "executor": executor,
    }


def check_create_install_lib(script, part, libs):
    if not (isinstance(libs, list) and libs):
        return

    params = get_platform_venv_params(script, part)

    install_libs = f"pip install {' '.join(libs)}"

    if not os.path.isdir(params["env_path"]):
        command = params["create_venv"]
        logger.info(command)

        process = subprocess.run(
            command,
            shell=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
        )
        logger.debug(process.stdout.splitlines())

    command = f"{params['activate_venv']} && {install_libs}"

    logger.info(command)

    process = subprocess.run(
        command,
        shell=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
    )

    logger.debug(process.stdout.splitlines())


def get_image_from_omero(a_task) -> str or None:
    image_id = a_task["omeroId"]
    file = FileManager(image_id)
    if file.exists():
        return file.get_filename()

    author = a_task.get("author").get("login")
    send_event(
      "omero/download/image", {"id": image_id, "override": False, "user": author}
    )
    return None


def run_subprocess(folder, script, part, data) -> dict:
    params = get_platform_venv_params(script, part)
    script_path = os.path.join(params["script_copy_path"], str(uuid.uuid4()))

    try:
        shutil.copytree(os.path.join(folder, part), script_path)
        runner_path = os.path.join(script_path, "__runner__.py")
        shutil.copyfile(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "runner.py"),
            runner_path,
        )

        filename = os.path.join(script_path, "__runner__.pickle")
        with open(filename, "wb") as infile:
            pickle.dump(data, infile)

        command = f"{params['activate_venv']} && {params['executor']} {runner_path}"
        logger.info(command)

        process = subprocess.run(
            command,
            shell=True,
            universal_newlines=True,
            capture_output=True,
            text=True,
        )
        hist_data = {
            "stderr": process.stderr if process.stderr else "",
            "stdout": process.stdout if process.stdout else "",
        }
        if process.stderr:
            logger.error(process.stderr)
        if process.stdout:
            logger.debug(process.stdout)

        with open(filename, "rb") as outfile:
            result_data = pickle.load(outfile)
            return {**data, **result_data, **hist_data}
    finally:
        shutil.rmtree(script_path, ignore_errors=True)


def start_scenario(
        script: str = "",
        part: str = "",
        folder: str = "",
        **kwargs,
):
    manifest = os.path.join(folder, part, "manifest.json")

    if not os.path.isfile(manifest):
        return None

    with open(manifest) as meta:
        data = json.load(meta)

    if not data:
        return None

    logger.info(f"{script}-{part}")
    params = data.get("params")
    for key, item in params.items():
        if kwargs.get(key) is None:
            raise ValueError(
                f"Not have param '{key}' in script: {script}, in part {part}"
            )

    check_create_install_lib(script, part, data.get("libs", []))
    return run_subprocess(folder, script, part, kwargs)


def get_pool_size(env_name) -> int:
    value = getenv(env_name, 'cpus')
    if value.lower() == 'cpus':
        value = cpu_count()
    return max(2, int(value))


def enrich_task_data(a_task):
    parent_jobs = db_instance().select(
        "pipeline_direction",
        "FILTER doc._to == @value",
        value=f"jobs/{a_task['parent']}",
    )

    if not parent_jobs:
        return {}

    data = {}
    jobs_ids = [item["_from"][5:] for item in parent_jobs]
    tasks = db_instance().select(
        "tasks",
        "FILTER doc.parent in @value "
        'and doc.result != "" '
        "and doc.result != Null ",
        value=jobs_ids,
    )

    for item in tasks:
        filename = getAbsoluteRelative(item["result"], True)
        with open(filename, "rb") as outfile:
            current_file_data = pickle.load(outfile)
            data = {**data, **current_file_data}

    return data


def update_status(status, a_task, result=None):
    search = "FILTER doc._key == @value LIMIT 1"
    data = {"status": status}

    if result:
        data.update({"result": result})
    if can_start(a_task["_id"]):
        db_instance().update(collection, data, search, value=a_task["id"])
        add_hist(a_task["_id"], f'status from: {a_task["status"]} to: {status}')
    else:
        data = {"status": -2}
        db_instance().update(collection, data, search, value=a_task["id"])


def get_path(job_id, task_id):
    path = os.path.join(os.getenv("DATA_STORAGE"), "jobs", job_id, task_id)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    return path


def get_task_with_status(_id: str):
    tasks = db_instance().select(
        collection,
        "FILTER doc._id == @id and doc.status == 1"
        "LIMIT 1 ",
        id=_id,
    )
    if len(tasks) == 1:
        return task(tasks[0]).to_json()
    else:
        return None


async def __executor(event):
    a_task = event.data.get("task")

    if not a_task:
        return

    a_task = a_task["_id"]

    if not a_task:
        return

    a_task = get_task_with_status(a_task)

    if not a_task:
        return

    update_status(2, a_task)
    a_task["params"] = {**a_task["params"], **enrich_task_data(a_task)}

    # download image tiff
    if not a_task["params"].get("image_path"):
        path = get_image_from_omero(a_task)
    else:
        path = a_task["params"].get("image_path")
        if not os.path.isfile(path):
            path = get_image_from_omero(a_task)

    if path is None:
        update_status(-1, a_task)
        return None

    script_path = getAbsoluteRelative(
        os.path.join(
            os.getenv("DATA_STORAGE"), "Scripts", f'{a_task["params"]["script"]}'
        )
    )

    filename = os.path.join(get_path(a_task["id"], a_task["parent"]), "result.pickle")

    if os.path.isfile(path):
        a_task["params"].update(image_path=path, folder=script_path)

        result = start_scenario(**a_task["params"])
        if not result:
            logger.info(f'problems with scenario params {a_task["params"]}')
        else:
            hist_dict = {
                key: result[key]
                for key in ('stderr', 'stdout')
            }
            add_hist(a_task["id"], hist_dict)

            result = {
                key: result[key]
                for key in result.keys() if key not in ('stderr', 'stdout')
            }

            with open(filename, "wb") as outfile:
                pickle.dump(result, outfile)

    if os.path.isfile(filename):
        update_status(100, a_task, result=getAbsoluteRelative(filename, False))
        logger.info("1 task complete")
    else:
        update_status(-1, a_task)
        logger.info("1 task uncompleted go to -1")


def worker(name):
    global logger

    logger = get_logger(name)
    redis_client = create_aioredis_client()

    @redis_client.event(EVENT_TYPE)
    async def listener(event):
        logger.debug(f'catch event: {event}')
        await __executor(event)

    try:
        logger.info('Starting')
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
        redis_client.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f'catch exception: {e}')
    finally:
        logger.info('Closing')
        redis_client.close()


class Worker(Process):
    def __init__(self, index=0):
        super().__init__(
            name=f'Spex.JM.Worker.{index + 1}',
            target=worker,
            args=(f'spex.ms-job-manager.worker.{index + 1}',),
            daemon=True
        )
