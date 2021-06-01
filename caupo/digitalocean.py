"""
Auxiliary module for working with Digitalocean's API in order to do architecture
changes easier than through the Dashboard
"""

import argparse
import logging
import os
import pprint

import requests

logger = logging.getLogger("caupo")

BASE_API_URL = "https://api.digitalocean.com"
TOKEN = os.environ.get("DO_TOKEN")


def list_droplets() -> None:
    """List information of all droplets accessible to the provided token"""

    response = requests.get(
        url=f"{BASE_API_URL}/v2/droplets",
        headers={
            "Authorization": f"Bearer {TOKEN}",
        }
    )
    logger.debug("Response: %s", response.json())
    pprint.pprint(response.json())


def power_off(droplet_id: str) -> None:
    """Power off a Digitalocean instance"""

    response = requests.post(
        url=f"{BASE_API_URL}/v2/droplets/{droplet_id}/actions",
        headers={
            "Authorization": f"Bearer {TOKEN}",
        },
        data={
            "type": "shutdown",
        }
    )
    logger.debug("Response: %s", response.json())
    pprint.pprint(response.json())


def power_on(droplet_id: str) -> None:
    """Power on a Digitalocean instance"""

    response = requests.post(
        url=f"{BASE_API_URL}/v2/droplets/{droplet_id}/actions",
        headers={
            "Authorization": f"Bearer {TOKEN}",
        },
        data={
            "type": "power_on",
        }
    )
    logger.debug("Response: %s", response.json())
    pprint.pprint(response.json())


def resize(droplet_id: str, size: str) -> None:
    """Resize a Digitalocean instance"""

    response = requests.post(
        url=f"{BASE_API_URL}/v2/droplets/{droplet_id}/actions",
        headers={
            "Authorization": f"Bearer {TOKEN}",
        },
        data={
            "type": "resize",
            "disk": False,
            "size": size,
        }
    )
    logger.debug("Response: %s", response.json())
    pprint.pprint(response.json())


def main() -> None:
    """Read input parameters and perform actions"""

    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["power_on", "power_off", "resize", "list_droplets"])
    parser.add_argument("--droplet_id", type=str, required=False)
    parser.add_argument("--size", type=str, required=False)
    args = parser.parse_args()

    logger.debug("You requested a `%s` action", args.action)

    if not TOKEN:
        logger.warning("No Digitalocean token was found! Please set the `DO_TOKEN` environment variable")
        return

    if args.action != "list_droplets" and args.droplet_id is None:
        logger.warning("You need to specify a `--droplet-id` for this action!")
        return
    else:
        logger.debug("Provided droplet ID: `%s`", args.droplet_id)

    if args.action == "resize":
        logger.debug("Requested size: `%s`", args.size)

    if args.action == "list_droplets":
        list_droplets()
    elif args.action == "resize":
        resize(args.droplet_id)
    elif args.action == "power_on":
        power_on(args.droplet_id)
    elif args.action == "power_off":
        power_off(args.droplet_id)


if __name__ == "__main__":
    main()
