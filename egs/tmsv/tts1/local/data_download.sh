#!/bin/bash -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
download_url=https://drive.google.com/open?id=1kyg_GuBZhxm8dRKTe6ioegMcSid9ZepV

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root_dir>"
    exit 1
fi

# download dataset
if [ ! -d ${db}/TMSV ]; then
    mkdir -p ${db}
    download_from_google_drive.sh ${download_url} ${db} zip
    echo "Successfully finished download."
else
    echo "Already exists. Skip download."
fi
