#!/bin/bash

# Copyright 2021 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Evaluation script for ground truth


echo "$0 $*"  # Print the command line for logging
. ./path.sh

nj=1
do_delta=false
db_root=""
backend=pytorch
api=v2
vocoder=
mcep_dim=24
shift_ms=5
asr_model="aishell.transformer.v1"
help_message="Usage: $0 <outdir> <subset> <srcspk> <trgspk>"

tmsv_spks=(
    "SP01" "SP02" "SP03" "SP04" "SP05" "SP06" "SP07" "SP08" "SP09" "SP10" "SP11" "SP12" "SP13" "SP14" "SP15" "SP16" "SP17" "SP18"
)

. utils/parse_options.sh

expdir=$1
name=$2  # dev or test

if [ $# != 2 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail
    
echo "step 0: Model preparation"
# ASR model selection for CER/WER objective evaluation 
asr_model_dir="exp/${asr_model}_asr"
    case "${asr_model}" in
        "aishell.transformer.v1") asr_url="https://drive.google.com/open?id=1BIQBpLRRy3XSMT5IRxnLcgLMirGzu8dg" \
            asr_cmvn="${asr_model_dir}/data/train_sp/cmvn.ark" \
            asr_pre_decode_config="${asr_model_dir}/conf/decode.yaml" \
            recog_model="${asr_model_dir}/exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/results/model.last10.avg.best" \
            lang_model="${asr_model_dir}/exp/train_rnnlm_pytorch_lm/rnnlm.model.best" ;;

    *) echo "No such models: ${asr_model}"; exit 1 ;;
esac

# ASR model download (librispeech)
if [ ! -e ${asr_model_dir}/.complete ]; then
    mkdir -p ${asr_model_dir}
    download_from_google_drive.sh ${asr_url} ${asr_model_dir} ".tar.gz"
    touch ${asr_model_dir}/.complete
fi
echo "ASR model: ${asr_model_dir} exits."

# setting dir
asr_data_dir="${expdir}/asr/data"
asr_fbank_dir="${expdir}/asr/fbank"
asr_feat_dir="${expdir}/asr/dump"
asr_result_dir="${expdir}/asr/result"

echo "step 1: Data preparation for ASR"
# Data preparation for ASR
local/ob_eval/data_prep_for_asr_vae.sh ${expdir} ${asr_data_dir}
text=${asr_data_dir}/text
[ -e ${text} ] && rm ${text}
for spk in "${tmsv_spks[@]}"; do
    cat data/${spk}_${name}/text >> ${text}.tmp
done
local/ob_eval/traditional2simplified.py --not_handle_utt2spk_uttid ${text}.tmp ${asr_data_dir}/utt2spk > ${text}
utils/validate_data_dir.sh --no-feats ${asr_data_dir}

echo "step 2: Feature extraction for ASR"
# Feature extraction for ASR
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} \
  --write_utt2num_frames true \
  --write_utt2dur false \
  ${asr_data_dir} \
  ${expdir}/fbank \
  ${asr_fbank_dir}

utils/fix_data_dir.sh ${asr_data_dir}

dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
  ${asr_data_dir}/feats.scp ${asr_cmvn} ${expdir}/dump_feats \
  ${asr_feat_dir}


echo "step 3: Dictionary and Json Data Preparation for ASR"
# Dictionary and Json Data Preparation for ASR
asr_dict="data/asr_dict/X.txt"; mkdir -p ${asr_dict%/*}
echo "<unk> 1" > ${asr_dict}

data2json.sh --feat ${asr_feat_dir}/feats.scp \
  ${asr_data_dir} ${asr_dict} > ${asr_feat_dir}/data.json


echo "step 4: ASR decoding"
# ASR decoding
asr_decode_config="conf/ob_eval/decode_asr.yaml"
cat < ${asr_pre_decode_config} | sed -e 's/beam-size: 60/beam-size: 10/' > ${asr_decode_config}

# split data
splitjson.py --parts ${nj} ${asr_feat_dir}/data.json

# set batchsize 0 to disable batch decoding    
${decode_cmd} JOB=1:${nj} ${asr_result_dir}.${api}/log/decode.JOB.log \
    asr_recog.py \
      --config ${asr_decode_config} \
      --ngpu 0 \
      --backend ${backend} \
      --batchsize 0 \
      --recog-json ${asr_feat_dir}/split${nj}utt/data.JOB.json \
      --result-label ${asr_result_dir}.${api}/data.JOB.json \
      --model ${recog_model} \
      --api ${api} \
      --rnnlm ${lang_model}

# calculate CER and show it.
local/ob_eval/score_sclite_wo_dict_modified.sh ${asr_result_dir}.${api}
