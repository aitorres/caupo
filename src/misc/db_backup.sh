#!/bin/bash

# Script that will make a dump of the database and store it into
# a given directory. Should be run periodically with cron and directed to
# a safe location in case backups need to be used / restored.

export DATE=`date '+%Y_%m_%d_%H_%M_%S'`

cd /mnt/volume_sfo3_01/backups
mkdir "$DATE"
cd "$DATE"
mongodump --host=127.0.0.1 --port=27019
