#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# text cleaner (Mandarin chars to IPA) for COSPRO

import argparse
import codecs
import re

import pinyin2ipa.pinyin_transform as pt
from pypinyin import lazy_pinyin, Style


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="text to be cleaned")
    args = parser.parse_args()

    style = Style.BOPOMOFO
    syllable_map = pt.SYLLABLE_MAP(source='Zhuyin', target="IPA")

    with codecs.open(args.text, "r", "utf-8") as fid:
        for line in fid.readlines():
            line = line.split()
            _id = line[0]
            text = "".join(line[1:])

            pinyin_list = lazy_pinyin(text, style=style)
            re_text = ''
            for s in pinyin_list:
                tone = pt.get_chewing_tone(s)
                syllable = re.sub("[ˊˇˋ˙]", "", s)
                ipa = syllable_map.transform(syllable).format(tone)
                re_text += ipa
                re_text += " "
            re_text = re_text[:-1] + '.'

            print("%s %s" % (_id, re_text))
