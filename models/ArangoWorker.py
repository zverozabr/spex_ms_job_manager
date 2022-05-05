from multiprocessing import Process
from spex_common.modules.logging import get_logger
from spex_common.modules.aioredis import send_event
from spex_common.services.Timer import every
from spex_common.modules.database import db_instance
from spex_common.models.Task import task
from spex_common.models.History import history
from datetime import datetime
import logging
EVENT_TYPE = 'backend/start_job'
collection = "tasks"


def add_hist(parent, content):
    db_instance().insert('history', history({
        'author': {'login': 'job_manager_catcher', 'id': '0'},
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


def worker(name):
    logger = get_logger(name)

    def listener():
        if a_task := get_task():
            update_status(1, a_task)
            send_event('backend/start_job', {"task": a_task})
            logger.info(f'founded task send it to in work: {a_task.get("name")} / {a_task.get("key")}')

    try:
        logger.info('Starting')
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
        every(5, listener)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f'catch exception: {e}')
    finally:
        logger.info('Closing')


class Worker(Process):
    def __init__(self, index=0):
        super().__init__(
            name=f'Spex.arango-job-catcher.Worker.{index + 1}',
            target=worker,
            args=(f'spex.ms-job-catcher.worker.{index + 1}',),
            daemon=True
        )
