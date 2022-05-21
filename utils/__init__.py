from datetime import datetime
from spex_common.modules.database import db_instance
from spex_common.models.History import history
from spex_common.models.WaitTableEntry import wait_table_entry


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


def add_to_waiting_table(login, who_waits, what_awaits):
    db_instance().insert('waiting_table', wait_table_entry({
        'author': {
            'login': login,
            'id': '0'
        },
        'date': str(datetime.now()),
        'who_waits': who_waits,
        'what_awaits': what_awaits,
    }).to_json())


def already_in_waiting_table(what_awaits):
    last_records = db_instance().select(
        'waiting_table',
        'FILTER doc.what_awaits == @what_awaits',
        what_awaits=what_awaits
    )
    key_arr = [record["_key"] for record in last_records]

    if not key_arr:
        return False

    return True


def del_from_waiting_list(what_awaits):
    last_records = db_instance().delete(
        'waiting_table',
        'FILTER doc.what_awaits == @what_awaits',
        what_awaits=what_awaits
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
        data = {"status": -2}
        db_instance().update(collection, data, search, value=a_task["id"])
