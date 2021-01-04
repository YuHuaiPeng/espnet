#!/usr/bin/env python3

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

import opencc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="original text in traditional Chinese")
    parser.add_argument("utt2spk", type=str, help="utt2spk")
    args = parser.parse_args()

    converter = opencc.OpenCC('t2s.json')
    
    # NOTE(unilight): because the original wav_text is not sorted,
    # we convert all sentences first, then print those in utt2spk.
    sim_sentences = {}
    with codecs.open(args.text, "r", "utf-8") as fid:
        for line in fid.readlines():
            line = line.split()
            _id = line[0]
            content = "".join(line[1:])
            sim_content = converter.convert(content)
            sim_sentences[_id] = sim_content

    # read utt2spk
    _ids = []
    with codecs.open(args.utt2spk, "r", "utf-8") as fid:
        for line in fid.readlines():
            line = line.split(" ")
            _id = line[0]
            if _id in sim_sentences:
                print("%s %s" % (_id, sim_sentences[_id]))
