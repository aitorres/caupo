"""
Auxiliary module with utility functions for managing the database
in certain processes
"""

import sys

from pymongo import MongoClient

mongo_client = MongoClient('mongodb://127.0.0.1:27019')
db = mongo_client.caupo_settings


def is_locked() -> None:
    """Determines whether the database is in a locked state"""

    return db.locking.find_one({"locked": True})


def lock() -> None:
    """Locks the database"""

    db.locking.insert_one({"locked": True})


def unlock() -> None:
    """Unlocks the database"""

    db.locking.delete_many({"locked": True})


def main() -> None:
    """Small script that un/locks the database on demand"""

    try:
        action = sys.argv[1]
    except IndexError:
        print(f"Usage:\tpython {sys.argv[0]} <action>")
        print("       \taction: check|lock|unlock")
        sys.exit(1)

    if action == "lock":
        lock()
    elif action == "unlock":
        unlock()
    else:
        state = "locked" if is_locked() else "unlocked"
        print(f"The database is currently {state}.")


if __name__ == "__main__":
    main()
