#!/bin/bash

# Copyright 2021 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=16        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000      # sampling frequency
fmax=7600     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# silence part trimming related
trim_threshold=30 # (in decibels)
trim_win_length=1024
trim_shift_length=256
trim_min_silence=0.01

# config files
train_config=conf/train_pytorch_transformer.tts_pt_cospro.one2many.yaml
decode_config=conf/decode.yaml

# decoding related
pwg_download_url=https://drive.google.com/open?id=150RjZuga8a5rc0opRr2oug5igkvlSUHE
outdir=                     # In case not evaluation not executed together with decoding & synthesis stage
model=                      # VC Model checkpoint for decoding. If not specified, automatically set to the latest checkpoint 
voc=PWG                      # vocoder used (GL or PWG)
voc_expdir=downloads/pwg
griffin_lim_iters=64        # The number of iterations of Griffin-Lim

# pretrained model related
pretrained_model_download_url=https://drive.google.com/open?id=1BxyZN8qJwi2kFGKfUOg36e8ilOvN2rr_
pretrained_model_path=downloads/cospro_tts_pt/exp/train_pytorch_ept_baseline/ept_results/snapshot.ep.100
cmvn=

# dataset configuration
tmsv_download_url=https://drive.google.com/open?id=1kyg_GuBZhxm8dRKTe6ioegMcSid9ZepV
dysarthric_download_url=
db_root=downloads
srcspk=dysarthric               
norm_name=cospro                # used to specify normalized data.

# objective evaluation related
mcep_dim=24
shift_ms=5

# exp tag
tag=""  # tag for managing experiments.

tmsv_spks=(
    "SP01" "SP02" "SP03" "SP04" "SP05" "SP06" "SP07" "SP08" "SP09" "SP10" "SP12" "SP13" "SP14" "SP15" "SP16" "SP17" "SP18"
)
all_spks=("${tmsv_spks[@]}")
all_spks+=("dysarthric")

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Resources download and preprocessing"

    local/data_download.sh ${db_root}/TMSV ${tmsv_download_url} zip
    local/data_download.sh ${db_root}/dysarthric ${dysarthric_download_url} zip
    local/data_download.sh ${db_root}/cospro_tts_pt ${pretrained_model_download_url} tar.gz
    local/data_download.sh ${db_root}/pwg ${pwg_download_url} zip

    # Power normalization
    local/normpow.sh ${db_root}/dysarthric/dysarthric ${db_root}/dysarthric/dysarthric_wav_normpow
    
    for spk in "${tmsv_spks[@]}"; do
        local/normpow.sh ${db_root}/TMSV/TMSV/${spk} ${db_root}/TMSV/${spk}_wav_normpow
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"

    rm -rf data/dysarthric
    local/data_prep.sh ${db_root}/dysarthric/dysarthric_wav_normpow dysarthric data/dysarthric
    utils/data/resample_data_dir.sh ${fs} data/dysarthric # Downsample to fs
    utils/fix_data_dir.sh data/dysarthric
    utils/validate_data_dir.sh --no-feats data/dysarthric

    for spk in "${tmsv_spks[@]}"; do
        rm -rf data/${spk}
        local/data_prep.sh ${db_root}/TMSV/${spk}_wav_normpow ${spk} data/${spk}
        utils/fix_data_dir.sh data/${spk}
        utils/validate_data_dir.sh --no-feats data/${spk}
    done
fi

if [ -z ${norm_name} ]; then
    echo "Please specify --norm_name ."
    exit 1
fi
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    if [ -z ${cmvn} ]; then
        echo "Please specify --cmvn ."
    fi

    fbankdir=fbank
    for spk in "${all_spks[@]}"; do
        echo "Generating fbanks features for ${spk}..."

        spk_train_set=${spk}_train
        spk_dev_set=${spk}_dev
        spk_eval_set=${spk}_eval
        spk_feat_tr_dir=${dumpdir}/${spk_train_set}_${norm_name}; mkdir -p ${spk_feat_tr_dir}
        spk_feat_dt_dir=${dumpdir}/${spk_dev_set}_${norm_name}; mkdir -p ${spk_feat_dt_dir}
        spk_feat_ev_dir=${dumpdir}/${spk_eval_set}_${norm_name}; mkdir -p ${spk_feat_ev_dir}

        # Trim silence parts at the begining and the end of audio
        mkdir -p exp/trim_silence/${spk}/figs  # avoid error
        trim_silence.sh --cmd "${train_cmd}" \
            --fs ${fs} \
            --win_length ${trim_win_length} \
            --shift_length ${trim_shift_length} \
            --threshold ${trim_threshold} \
            --min_silence ${trim_min_silence} \
            data/${spk} \
            exp/trim_silence/${spk}

        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${spk} \
            exp/make_fbank/${spk}_${norm_name} \
            ${fbankdir}

        # make train/dev/eval set
        utils/subset_data_dir.sh --last data/${spk} 80 data/${spk}_tmp
        utils/subset_data_dir.sh --last data/${spk}_tmp 40 data/${spk_eval_set}
        utils/subset_data_dir.sh --first data/${spk}_tmp 40 data/${spk_dev_set}
        n=$(( $(wc -l < data/${spk}/wav.scp) - 80 ))
        utils/subset_data_dir.sh --first data/${spk} ${n} data/${spk_train_set}
        rm -rf data/${spk}_tmp

        # dump features
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${spk_train_set}/feats.scp ${cmvn} exp/dump_feats/${spk_train_set}_${norm_name} ${spk_feat_tr_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${spk_dev_set}/feats.scp ${cmvn} exp/dump_feats/${spk_dev_set}_${norm_name} ${spk_feat_dt_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${spk_eval_set}/feats.scp ${cmvn} exp/dump_feats/${spk_eval_set}_${norm_name} ${spk_feat_ev_dir}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"

    # make dummy dict
    dict="data/dummy_dict/X.txt"
    mkdir -p ${dict%/*}
    echo "<unk> 1" > ${dict}

    # make json labels
    for spk in "${all_spks[@]}"; do

        spk_train_set=${spk}_train
        spk_dev_set=${spk}_dev
        spk_eval_set=${spk}_eval
        spk_feat_tr_dir=${dumpdir}/${spk_train_set}_${norm_name}
        spk_feat_dt_dir=${dumpdir}/${spk_dev_set}_${norm_name}
        spk_feat_ev_dir=${dumpdir}/${spk_eval_set}_${norm_name}

        data2json.sh --feat ${spk_feat_tr_dir}/feats.scp \
             data/${spk_train_set} ${dict} > ${spk_feat_tr_dir}/data.json
        data2json.sh --feat ${spk_feat_dt_dir}/feats.scp \
             data/${spk_dev_set} ${dict} > ${spk_feat_dt_dir}/data.json
        data2json.sh --feat ${spk_feat_ev_dir}/feats.scp \
             data/${spk_eval_set} ${dict} > ${spk_feat_ev_dir}/data.json
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: x-vector extraction"
    
    # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    
    mfccdir=mfcc
    vaddir=mfcc
    for spk in "${tmsv_spks[@]}"; do
        echo "Extracting x-vector for ${spk}"
        train_set=${spk}_train
        dev_set=${spk}_dev
        eval_set=${spk}_eval

        # Make MFCCs and compute the energy-based VAD for each dataset
        for name in ${train_set} ${dev_set} ${eval_set}; do
            utils/copy_data_dir.sh data/${name} data/${name}_mfcc_16k
            utils/data/resample_data_dir.sh 16000 data/${name}_mfcc_16k
            steps/make_mfcc.sh \
                --write-utt2num-frames true \
                --mfcc-config conf/mfcc.conf \
                --nj 1 --cmd "$train_cmd" \
                data/${name}_mfcc_16k exp/make_mfcc_16k ${mfccdir}
            utils/fix_data_dir.sh data/${name}_mfcc_16k
            sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
                data/${name}_mfcc_16k exp/make_vad ${vaddir}
            utils/fix_data_dir.sh data/${name}_mfcc_16k
        done

        # Extract x-vector
        for name in ${train_set} ${dev_set} ${eval_set}; do
            sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 1 \
                ${nnet_dir} data/${name}_mfcc_16k \
                ${nnet_dir}/xvectors_${name}
        done
        # Update json
        for name in ${train_set} ${dev_set} ${eval_set}; do
            local/update_json.sh ${dumpdir}/${name}_${norm_name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
        done
    done
fi

pair=${srcspk}_all
pair_train_set=${pair}_train
pair_dev_set=${pair}_dev
pair_eval_set=${pair}_eval
pair_tr_dir=${dumpdir}/${pair_train_set}_${norm_name}; mkdir -p ${pair_tr_dir}
pair_dt_dir=${dumpdir}/${pair_dev_set}_${norm_name}; mkdir -p ${pair_dt_dir}
pair_ev_dir=${dumpdir}/${pair_eval_set}_${norm_name}; mkdir -p ${pair_ev_dir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Merge Pair Json Data"
    
    train_data_jsons=""
    dev_data_jsons=""
    eval_data_jsons=""
    for spk in "${tmsv_spks[@]}"; do
        train_data_jsons+=" ${dumpdir}/${spk}_train_${norm_name}/data.json"
        dev_data_jsons+=" ${dumpdir}/${spk}_dev_${norm_name}/data.json"
        test_data_jsons+=" ${dumpdir}/${spk}_eval_${norm_name}/data.json"
    done

    local/merge_pair_json.py \
        --src-json ${dumpdir}/${srcspk}_train_${norm_name}/data.json \
        -O ${pair_tr_dir}/data.json \
        ${train_data_jsons}
    local/merge_pair_json.py \
        --src-json ${dumpdir}/${srcspk}_dev_${norm_name}/data.json \
        -O ${pair_dt_dir}/data.json \
        ${dev_data_jsons}
    local/merge_pair_json.py \
        --src-json ${dumpdir}/${srcspk}_eval_${norm_name}/data.json \
        -O ${pair_ev_dir}/data.json \
        ${test_data_jsons}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: VC model training"

    if [[ -z ${train_config} ]]; then
        echo "Please specify --train_config."
        exit 1
    fi

    # If pretrained model specified, add pretrained model info in config
    if [ -n "${pretrained_model_path}" ]; then
        train_config="$(change_yaml.py \
            -a enc-init="${pretrained_model_path}" \
            -a dec-init="${pretrained_model_path}" \
            -o "conf/$(basename "${train_config}" .yaml).${tag}.yaml" "${train_config}")"
    fi
    if [ -z ${tag} ]; then
        expname=${srcspk}_all_${backend}_$(basename ${train_config%.*})
    else
        expname=${srcspk}_all_${backend}_${tag}
    fi
    expdir=exp/${expname}

    mkdir -p ${expdir}
    tr_json=${pair_tr_dir}/data.json
    dt_json=${pair_dt_dir}/data.json

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Decoding and synthesis"
    
    if [ -z ${cmvn} ]; then
        echo "Please specify --cmvn ."
        exit 1
    fi

    if [ -z ${tag} ]; then
        expname=${srcspk}_all_${backend}_$(basename ${train_config%.*})
    else
        expname=${srcspk}_all_${backend}_${tag}
    fi
    expdir=exp/${expname}

    if [ -z "${model}" ]; then
        model="$(find "${expdir}" -name "snapshot*" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
        model=$(basename ${model})
    fi
    outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})

    echo "Decoding..."
    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}_${norm_name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    echo "Synthesis..."

    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}

        # Normalization
        # If not using pretrained models statistics, use statistics of target speaker
        if [ -n "${cmvn}" ]; then
            trg_cmvn="${cmvn}"
        else
            trg_cmvn=data/${trg_train_set}/cmvn.ark
        fi
        apply-cmvn --norm-vars=true --reverse=true ${trg_cmvn} \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp

        # GL
        if [ ${voc} = "GL" ]; then
            echo "Using Griffin-Lim phase recovery."
            convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                --iters ${griffin_lim_iters} \
                ${outdir}_denorm/${name} \
                ${outdir}_denorm/${name}/log \
                ${outdir}_denorm/${name}/wav
        # PWG
        elif [ ${voc} = "PWG" ]; then
            echo "Using Parallel WaveGAN vocoder."

            # check existence
            if [ ! -d ${voc_expdir} ]; then
                echo "${voc_expdir} does not exist. Please download the pretrained model."
                exit 1
            fi

            # variable settings
            voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t | tail -n +1 | head -n 1)"
            voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | tail -n +1 | head -n 1)"
            voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | tail -n +1 | head -n 1)"
            wav_dir=${outdir}_denorm/${name}/pwg_wav
            hdf5_norm_dir=${outdir}_denorm/${name}/hdf5_norm
            [ ! -e "${wav_dir}" ] && mkdir -p ${wav_dir}
            [ ! -e ${hdf5_norm_dir} ] && mkdir -p ${hdf5_norm_dir}

            # normalize and dump them
            echo "Normalizing..."
            ${train_cmd} "${hdf5_norm_dir}/normalize.log" \
                parallel-wavegan-normalize \
                    --skip-wav-copy \
                    --config "${voc_conf}" \
                    --stats "${voc_stats}" \
                    --feats-scp "${outdir}_denorm/${name}/feats.scp" \
                    --dumpdir ${hdf5_norm_dir} \
                    --verbose "${verbose}"
            echo "successfully finished normalization."

            # decoding
            echo "Decoding start. See the progress via ${wav_dir}/decode.log."
            ${cuda_cmd} --gpu 1 "${wav_dir}/decode.log" \
                parallel-wavegan-decode \
                    --dumpdir ${hdf5_norm_dir} \
                    --checkpoint "${voc_checkpoint}" \
                    --outdir ${wav_dir} \
                    --verbose "${verbose}"

            # renaming
            rename -f "s/_gen//g" ${wav_dir}/*.wav

            echo "successfully finished decoding."
        else
            echo "Vocoder type not supported. Only GL and PWG are available."
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Objective Evaluation: MCD"
    
    for name in ${pair_dev_set} ${pair_eval_set}; do
        echo "Calculating MCD for ${name} set"

        all_mcds_file=${outdir}_denorm/${name}/all_mcd.log; rm -f ${all_mcds_file}
        for spk in "${tmsv_spks[@]}"; do
            out_wavdir=${outdir}_denorm/${name}/pwg_wav
            minf0=$(grep ${spk} conf/all.f0 | cut -f 2 -d" ")
            maxf0=$(grep ${spk} conf/all.f0 | cut -f 3 -d" ")
            out_spk_wavdir=${outdir}_denorm/mcd/${name}/pwg_out/${spk}
            mkdir -p ${out_spk_wavdir}
           
            # copy wav files for mcd calculation
            for out_wav_file in $(find -L ${out_wavdir} -iname "${spk}_*" | sort ); do
                wav_basename=$(basename $out_wav_file .wav)
                cp ${out_wav_file} ${out_spk_wavdir} || exit 1
            done

            # actual calculation
            mcd_file=${outdir}_denorm/${name}/${spk}_mcd.log
            ${decode_cmd} ${mcd_file} \
                local/mcd_calculate.py \
                    --wavdir ${out_spk_wavdir} \
                    --gtwavdir ${db_root}/TMSV/${spk}_wav_normpow \
                    --mcep_dim ${mcep_dim} \
                    --shiftms ${shift_ms} \
                    --f0min ${minf0} \
                    --f0max ${maxf0}
            grep "Mean MCD" < ${mcd_file} >> ${all_mcds_file}
            echo "${spk}: $(grep 'Mean MCD' < ${mcd_file})"
        done
        echo "Mean MCD for ${name} set is $(awk '{ total += $3; count++ } END { print total/count }' ${all_mcds_file})"
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Objective Evaluation: ASR"

    for name in ${pair_dev_set} ${pair_eval_set}; do
        local/ob_eval/evaluate_all_asr.sh --nj ${nj} \
            --db_root ${db_root} \
            --vocoder ${voc} \
            ${outdir} ${name}
    done
fi
