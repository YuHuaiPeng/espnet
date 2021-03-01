#!/usr/bin/env python3

# Copyright 2021 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="text to be cleaned")
    parser.add_argument("utt2spk", type=str, help="utt2spk file to extract id from")
    args = parser.parse_args()
    with codecs.open(args.text, "r", "utf-8") as textid, \
        codecs.open(args.utt2spk, "r", "utf-8") as utt2spkid:
        for textline, utt2spkline in zip(textid.read().splitlines(), utt2spkid.read().splitlines()):
            _id = utt2spkline.split(" ")[0]
            print("%s %s" % (_id, textline))
