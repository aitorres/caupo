"""
Auxiliary module with utility functions for managing the database
in certain processes
"""

import sys
from typing import Any, Optional

import numpy as np
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


def is_locked() -> Optional[Any]:
    """Determines whether the database is in a locked state"""

    return settings_db.locking.find_one({"locked": True})


def lock() -> None:
    """Locks the database"""

    settings_db.locking.insert_one({"locked": True})


def unlock() -> None:
    """Unlocks the database"""

    settings_db.locking.delete_many({"locked": True})


def transform_types_for_database(obj: Any) -> Any:
    """Given an object, transforms its type to a native Python type for storage in database"""

    # Numpy types
    if isinstance(obj, (np.generic)):
        return obj.item()

    # Collection types
    if isinstance(obj, (list, set)):
        return [transform_types_for_database(item) for item in obj]

    if isinstance(obj, np.ndarray):
        return transform_types_for_database(obj.tolist())

    # Structures
    if isinstance(obj, dict):
        return {key: transform_types_for_database(value) for key, value in obj.items()}

    return obj


def get_results_collection() -> pymongo.collection.Collection:
    """Returns the appropriate collection where results information should be stored"""

    return db.results


def result_already_exists(frequency: str, tag: str, algorithm: str, embedder: str) -> bool:
    """Given data of a result output, determines if the result already exists in the database"""

    collection = get_results_collection()
    return collection.find_one({
        'frequency': frequency,
        'tag': tag,
        'algorithm': algorithm,
        'embedder': embedder,
    })


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
