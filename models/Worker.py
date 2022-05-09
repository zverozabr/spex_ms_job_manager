import shutil
import os
import uuid
import json
import pickle
import subprocess
import logging
from os import cpu_count, getenv
from functools import partial
from multiprocessing import Process

from spex_common.models.Task import task
from spex_common.modules.logging import get_logger
from spex_common.modules.aioredis import create_aioredis_client
from spex_common.modules.database import db_instance
from spex_common.services.Utils import getAbsoluteRelative
from spex_common.modules.aioredis import send_event
from spex_common.models.OmeroImageFileManager import OmeroImageFileManager

from Constants import EVENT_TYPE, collection
from utils import (
    add_history as add_history_original,
    update_status as update_status_original
)

add_history = partial(add_history_original, 'job_manager_runner')
update_status = partial(update_status_original, collection, 'job_manager_runner')


def get_platform_venv_params(script, part):
    env_path = os.getenv("SCRIPTS_ENVS_PATH", f"~/scripts_envs")
    env_path = os.path.expanduser(env_path)

    script_copy_path = os.path.join(env_path, "scripts", script, part)
    os.makedirs(script_copy_path, exist_ok=True)

    env_path = os.path.join(env_path, "envs", script)
    os.makedirs(env_path, exist_ok=True)
    env_path = os.path.join(env_path, part)

    not_posix = os.name != "posix"

    executor = "python" if not_posix else "python3"
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


def get_image_from_omero(a_task) -> str or None:
    image_id = a_task["omeroId"]
    file = OmeroImageFileManager(image_id)
    if file.exists():
        return file.get_filename()

    author = a_task.get("author").get("login")
    send_event(
      "omero/download/image", {"id": image_id, "override": False, "user": author}
    )
    return None


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

    return task(tasks[0]).to_json() if len(tasks) == 1 else None


class Executor:
    def __init__(self, logger, task_id):
        self.logger = logger
        self.task_id = task_id

    def run(self):
        if not self.task_id:
            return

        a_task = get_task_with_status(self.task_id)

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
                os.getenv("DATA_STORAGE"),
                "Scripts",
                f'{a_task["params"]["script"]}'
            )
        )

        filename = os.path.join(get_path(a_task["id"], a_task["parent"]), "result.pickle")

        if os.path.isfile(path):
            a_task["params"].update(image_path=path, folder=script_path)

            result = self.start_scenario(**a_task["params"])
            if not result:
                self.logger.info(f'problems with scenario params {a_task["params"]}')
            else:
                hist_dict = {
                    key: result[key]
                    for key in ('stderr', 'stdout')
                }
                add_history(a_task["id"], hist_dict)

                result = {
                    key: result[key]
                    for key in result.keys() if key not in ('stderr', 'stdout')
                }

                with open(filename, "wb") as outfile:
                    pickle.dump(result, outfile)

        if os.path.isfile(filename):
            update_status(100, a_task, result=getAbsoluteRelative(filename, False))
            self.logger.info("1 task complete")
        else:
            update_status(-1, a_task)
            self.logger.info("1 task uncompleted go to -1")

    def start_scenario(
        self,
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

        self.logger.info(f"{script}-{part}")
        params = data.get("params")
        for key, item in params.items():
            if kwargs.get(key) is None:
                raise ValueError(
                    f"Not have param '{key}' in script: {script}, in part {part}"
                )

        self.check_create_install_lib(script, part, data.get("libs", []))
        return self.run_subprocess(folder, script, part, kwargs)

    def check_create_install_lib(self, script, part, libs):
        if not (isinstance(libs, list) and libs):
            return

        params = get_platform_venv_params(script, part)

        install_libs = f"pip install {' '.join(libs)}"

        if not os.path.isdir(params["env_path"]):
            command = params["create_venv"]
            self.logger.info(command)

            process = subprocess.run(
                command,
                shell=True,
                universal_newlines=True,
                stdout=subprocess.PIPE,
            )
            self.logger.debug(process.stdout.splitlines())

        command = f"{params['activate_venv']} && {install_libs}"

        self.logger.info(command)

        process = subprocess.run(
            command,
            shell=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
        )

        self.logger.debug(process.stdout.splitlines())

    def run_subprocess(self, folder, script, part, data) -> dict:
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
            self.logger.info(command)

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
                self.logger.error(process.stderr)
            if process.stdout:
                self.logger.debug(process.stdout)

            with open(filename, "rb") as outfile:
                result_data = pickle.load(outfile)
                return {**data, **result_data, **hist_data}
        finally:
            shutil.rmtree(script_path, ignore_errors=True)


async def __executor(logger, event):
    a_task = event.data.get("task")

    if not a_task:
        return

    a_task = a_task["_id"]

    if not a_task:
        return

    executor = Executor(logger, a_task)

    executor.run()


def worker(name):
    logger = get_logger(name)
    redis_client = create_aioredis_client()

    @redis_client.event(EVENT_TYPE)
    async def listener(event):
        if event is None or event.is_viewed:
            return
        logger.debug(f'catch event: {event}')
        event.set_is_viewed()
        await __executor(logger, event)

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
