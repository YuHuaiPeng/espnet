#!/bin/bash -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*"  # Print the command line for logging

db=$1
spk=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <spk> <data_dir>"
    exit 1
fi

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${text} ] && rm ${text}

# make scp, utt2spk, and spk2utt
find ${db} -name "*.wav" -follow | sort | while read -r filename;do
    id="${spk}_$(basename ${filename} | sed -e "s/\.[^\.]*$//g" | sed -e "s/_//g" | sed -e "s/-//g" | sed -e "s/${spk}//g" | sed -e "s/T//g")"
    echo "${id} ${filename}" >> ${scp}
    echo "${id} ${spk}" >> ${utt2spk}
done
echo "Successfully finished making wav.scp, utt2spk."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

# make text
local/clean_text.py downloads/TMSV/TMSV/tmhint.txt ${utt2spk} > ${text}
echo "Successfully finished making text."
