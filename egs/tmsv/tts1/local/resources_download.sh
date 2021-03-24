#!/bin/bash -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1
download_url="https://drive.google.com/open?id=1keikpLu1U75dZQPMtmaTbBcDawxk3D7T"

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    echo ""
    exit 1
fi

if [ ! -e ${download_dir}/.resources_complete ]; then
    download_from_google_drive.sh ${download_url} ${download_dir} ".tar.gz"
    mv ${download_dir}/downloads ${download_dir}/resources
    touch ${download_dir}/.resources_complete
fi
echo "Successfully finished donwload of resources."
