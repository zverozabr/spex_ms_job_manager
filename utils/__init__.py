import spex_common.services.Task as TaskService
from datetime import datetime
from spex_common.modules.database import db_instance
from spex_common.models.History import history
from spex_common.models.WaitTableEntry import wait_table_entry, WaitTableEntry
from spex_common.models.Status import TaskStatus


def add_history(login, parent, content):
    db_instance().insert('history', history({
        'author': {
            'login': login,
            'id': '0'
        },
        'date': str(datetime.now()),
        'content': content,
        'parent': parent,
    }).to_json())


def add_to_waiting_table(login, waiter_type, waiter_id, what_awaits):
    db_instance().insert('waiting_table', wait_table_entry({
        'author': {
            'login': login,
            'id': '0'
        },
        'date': str(datetime.now()),
        'waiter_id': waiter_id,
        'waiter_type': waiter_type,
        'what_awaits': what_awaits,
    }).to_json())


def already_in_waiting_table(what_awaits, waiter_type, waiter_id):
    last_records = db_instance().select(
        'waiting_table',
        f'FILTER doc.what_awaits == @what_awaits'
        f' and doc.waiter_type == @waiter_type'
        f' and doc.waiter_id == @waiter_id',
        what_awaits=what_awaits,
        waiter_type=waiter_type,
        waiter_id=waiter_id,
    )
    key_arr = [record["_key"] for record in last_records]

    if not key_arr:
        return False

    return True


def get_from_waiting_table(what_awaits, waiter_type) -> [WaitTableEntry]:
    last_records = db_instance().select(
        'waiting_table',
        'FILTER doc.what_awaits == @what_awaits and doc.waiter_type == @waiter_type',
        what_awaits=what_awaits,
        waiter_type=waiter_type,
    )

    return [wait_table_entry(item) for item in last_records] \
        if last_records \
        else []


def del_from_waiting_table(ids):
    db_instance().delete(
        'waiting_table',
        'FILTER doc._key in @ids',
        ids=ids
    )


def can_start(task_id):
    last_records = db_instance().select(
        "history",
        "FILTER doc.parent == @value SORT doc.date DESC LIMIT 3 ",
        value=task_id,
    )
    key_arr = [record["_key"] for record in last_records]
    if not key_arr:
        return True
    last_canceled_records = db_instance().select(
        "history",
        "FILTER doc.parent == @value"
        " and (doc.content Like @content "
        " or doc.content Like @content2) "
        "SORT doc.date DESC LIMIT 3 ",
        value=task_id,
        content="%-1 to: 1%",
        content2="%1 to: 2%"
    )
    key_arr_2 = [record["_key"] for record in last_canceled_records]
    return not (key_arr_2 == key_arr and len(key_arr) == 3)


def get_task_with_status(_id: str, status: int):
    tasks = TaskService.select_tasks(
        search="FILTER doc._id == @value and doc.status == @status LIMIT 1",
        value=_id,
        status=status
    )

    return tasks[0] if tasks else None


def get_tasks(ids):
    return TaskService.select_tasks(condition='in', _key=ids)


def update_status(collection, login, status, a_task, result=None):
    search = "FILTER doc._key == @value LIMIT 1"
    data = {"status": status}

    if result:
        data.update({"result": result})
    if can_start(a_task["_id"]):
        db_instance().update(collection, data, search, value=a_task["id"])
        add_history(
            login,
            a_task["_id"],
            f'status from: {a_task["status"]} to: {status}'
        )
    else:
        data = {"status": TaskStatus.pending_approval.value}
        db_instance().update(collection, data, search, value=a_task["id"])
