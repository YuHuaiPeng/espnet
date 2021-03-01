#!/bin/bash -e

# Copyright 2021 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*"  # Print the command line for logging

srcdb=$1
trgdb=$2

# remove and make directory
rm -rf ${trgdb}; mkdir -p ${trgdb}

# make scp, utt2spk, and spk2utt
find ${srcdb} -name "*.wav" -follow | sort | while read -r filename;do
    wavname=$(basename ${filename})
    trgwavname=${trgdb}/${wavname}
    tmp_trgwavname=${trgdb}/tmp_${wavname}

    # stereo to mono
    sox ${filename} -c 1 ${tmp_trgwavname}

    # normpow
    local/normpow/normpow/normpow ${tmp_trgwavname} ${trgwavname}
    rm -f ${tmp_trgwavname}
done
echo "Successfully finished normpow."
