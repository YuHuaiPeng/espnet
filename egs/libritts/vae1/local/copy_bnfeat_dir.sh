#!/bin/bash

echo "$0 $*" >&2 # Print the command line for logging
. ./path.sh

nj=1
cmd=run.pl

compress=true
write_utt2num_frames=false

suffix=_hires

help_message=$(cat << EOF
Usage: $0 <data-dir> <target-dir>
EOF
)
. utils/parse_options.sh

if [ $# != 3 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

name=$1
data_dir=$2
target_dir=$3

[ -f $nj ] && nj=`cat $nj`;

if $write_utt2num_frames; then
    write_num_frames_opt="--write-num-frames=ark,t:$logdir/utt2num_frames.JOB"
else
    write_num_frames_opt=
fi

logdir=$target_dir/log
mkdir -p $logdir

$cmd JOB=1:$nj $logdir/copy_bnfeats.JOB.log \
    copy-feats --compress=$compress $write_num_frames_opt \
        scp:$data_dir/raw_bnfeat_${name}${suffix}.JOB.scp \
        ark,scp:$target_dir/raw_bnfeat_${name}.JOB.ark,$target_dir/raw_bnfeat_${name}.JOB.scp

if [ -f $logdir/.error.$name ]; then
    echo "$0: Error copying bottleneck features for $name:"
    tail $logdir/make_fbank_${name}.1.log
    exit 1;
fi

for n in $(seq $nj); do
    cat $target_dir/raw_bnfeat_${name}.$n.scp || exit 1
done > $target_dir/bnfeats.scp

if $write_utt2num_frames; then
    for n in $(seq $nj); do
        cat $logdir/utt2num_frames.$n || exit 1
    done > $target_dir/utt2num_frames || exit 1
fi

echo "$0: Succeeded copying bottleneck features for $name"
