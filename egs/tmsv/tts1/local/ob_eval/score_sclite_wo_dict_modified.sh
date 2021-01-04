#!/bin/bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

wer=false
num_spkrs=1
help_message="Usage: $0 <data-dir>"

. utils/parse_options.sh

if [ $# != 1 ]; then
    echo "${help_message}"
    exit 1;
fi

dir=$1

concatjson.py ${dir}/data.*.json > ${dir}/data.json

if [ $num_spkrs -eq 1 ]; then
    json2trn_wo_dict.py ${dir}/data.json --num-spkrs ${num_spkrs} --refs ${dir}/ref_org.wrd.trn --hyps ${dir}/hyp_org.wrd.trn
  
    # NOTE(unilight): not really sure if sed is needed here
    cat < ${dir}/hyp_org.wrd.trn | sed -e 's/▁//' | sed -e 's/▁/ /g' > ${dir}/hyp.wrd.trn
    cat < ${dir}/ref_org.wrd.trn > ${dir}/ref.wrd.trn

    # awk will make chinese character produce some bug. We instead use a python script to replace it
    python local/ob_eval/make_trn_from_wrd-trn.py -i ${dir}/hyp.wrd.trn -o ${dir}/hyp.trn
    python local/ob_eval/make_trn_from_wrd-trn.py -i ${dir}/ref.wrd.trn -o ${dir}/ref.trn

    sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn -i rm -o all stdout > ${dir}/result.txt
    echo "write a CER result in ${dir}/result.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.txt
    
    python local/ob_eval/make_pinyin-trn_from_wrd-trn.py -i ${dir}/hyp.wrd.trn -o ${dir}/hyp.py.trn
    python local/ob_eval/make_pinyin-trn_from_wrd-trn.py -i ${dir}/ref.wrd.trn -o ${dir}/ref.py.trn

    sclite -r ${dir}/ref.py.trn trn -h ${dir}/hyp.py.trn -i rm -o all stdout > ${dir}/result.py.txt
    echo "write a Syllable Error Rate result in ${dir}/result.py.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.py.txt
fi


