#!/bin/bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <db> <data_dir>"
  exit 1
fi

db=$1
data_dir=$2

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


# make scp, utt2spk, and spk2utt
find ${db}/wav -name "*.wav" -follow | sort | while read -r filename;do
    id="$(basename ${filename} | sed -e "s/\.[^\.]*$//g")"
    spk="$(basename $(dirname ${filename}))"
    echo "${id} ${filename}" >> ${scp}
    echo "${id} ${spk}" >> ${utt2spk}
done
echo "Successfully finished making wav.scp, utt2spk."

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

sort ${db}/wav_text > ${text}_original
local/clean_text.py ${text}.original > ${text}
echo "Successfully finished making text."
exit 0
