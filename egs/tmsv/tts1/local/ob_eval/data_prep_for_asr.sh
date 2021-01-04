#!/bin/bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
# Modified Copyright 2020 Academia Sinica (Pin-Jui Ku)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

wav_dir=$1
data_dir=$2
set_name=$3

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}

# make scp
find ${wav_dir} -name "*.wav" | sort | while read -r filename;do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g" | sed -e "s/-feats//g" | sed -e "s/_gen//g"  )
    spk=$(echo "$id" | cut -f 1 -d"_")
    echo "${id} ${filename}" >> ${scp}
    echo "${id} $spk" >> ${utt2spk}
done

utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."
