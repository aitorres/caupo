"""
Auxiliary module with utility functions for managing the database
in certain processes
"""

import sys

import pymongo

client = pymongo.MongoClient('mongodb://127.0.0.1:27019')
db = client.caupo
settings_db = client.caupo_settings


def get_db() -> pymongo.database.Database:
    """Returns the database instance to be used accross the project for data storage"""

    return db


def get_settings_db() -> pymongo.database.Database:
    """Returns the database instance to be used accross the project for setting storage"""

    return settings_db


def is_locked() -> None:
    """Determines whether the database is in a locked state"""

    return settings_db.locking.find_one({"locked": True})


def lock() -> None:
    """Locks the database"""

    settings_db.locking.insert_one({"locked": True})


def unlock() -> None:
    """Unlocks the database"""

    settings_db.locking.delete_many({"locked": True})


def main() -> None:
    """Small script that un/locks the database on demand"""

    try:
        action = sys.argv[1]
    except IndexError:
        print(f"Usage:\tpython {sys.argv[0]} <action>")
        print("       \taction: check|lock|unlock")
        sys.exit(1)

    if action == "lock":
        # lock it lock it lock it lock it
        lock()
    elif action == "unlock":
        # got the key, can you unlock it?
        unlock()
    else:
        state = "locked" if is_locked() else "unlocked"
        print(f"The database is currently {state}.")


if __name__ == "__main__":
    main()
