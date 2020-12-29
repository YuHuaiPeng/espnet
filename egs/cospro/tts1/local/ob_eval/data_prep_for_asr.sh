#!/bin/bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
# Modified Copyright 2020 Academia Sinica (Pin-Jui Ku)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

wav_dir=$1
data_dir=$2
set_name=$3
db_root=$4    # NOTE(unilight): very dirty hack for text preparation

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${text} ] && rm ${text}
[ -e ${utt2spk} ] && rm ${utt2spk}

# make scp
find ${wav_dir} -name "*.wav" | sort | while read -r filename;do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g" | sed -e "s/-feats//g" | sed -e "s/_gen//g"  )
    echo "${id} ${filename}" >> ${scp}
done

# directly copy utt2spk since speaker is hard to parse due to strange naming rule of COSPRO dataset
cp data/${set_name}/utt2spk ${utt2spk}
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "Successfully finished making spk2utt."

# convert traditional Chiniese to simplified Chinese
# the text in data/ is IPA, so we use `wav_text` from the original dataset
local/ob_eval/traditional2simplified.py ${db_root}/wav_text data/${set_name}/utt2spk > ${text}
