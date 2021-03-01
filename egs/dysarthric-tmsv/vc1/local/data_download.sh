#!/bin/bash -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
download_url=$2
ext=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db_root_dir> <download_url> <ext (zip, tar.gz, etc)"
    exit 1
fi

# download dataset
if [ ! -d ${db} ]; then
    mkdir -p ${db}
    download_from_google_drive.sh ${download_url} ${db} ${ext}
    echo "Successfully finished download."
else
    echo "Already exists. Skip download."
fi
