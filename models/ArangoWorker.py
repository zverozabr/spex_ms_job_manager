import logging
from multiprocessing import Process
from functools import partial

from spex_common.models.Task import task
from spex_common.modules.aioredis import send_event
from spex_common.modules.database import db_instance
from spex_common.modules.logging import get_logger
from spex_common.services.Timer import every

from Constants import collection, EVENT_TYPE
from utils import (
    add_history as add_history_original,
    update_status as update_status_original
)

add_history = partial(add_history_original, 'job_manager_catcher')
update_status = partial(update_status_original, collection, 'job_manager_catcher')


def get_task():
    tasks = db_instance().select(
        collection,
        "FILTER doc.status == 0 or doc.status == -1 and doc.content like @value "
        "LIMIT 1 ",
        value="%empty%",
    )

    if len(tasks) == 1:
        return task(tasks[0]).to_json()

    tasks = db_instance().select(collection, " FILTER doc.status == 0 LIMIT 1  ")
    return task(tasks[0]).to_json() if len(tasks) == 1 else None


def worker(name):
    logger = get_logger(name)

    def listener():
        if a_task := get_task():
            update_status(1, a_task)
            send_event(EVENT_TYPE, {"task": a_task})
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
