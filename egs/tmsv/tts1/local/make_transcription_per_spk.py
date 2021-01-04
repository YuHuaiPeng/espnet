#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os
import sys

import pinyin2ipa.pinyin_transform as pt
import pypinyin
from pypinyin import lazy_pinyin, Style
import re


def get_the_utt_id(num, spk_id):
    num = num + 1  # The uttid starts from 1, so we add 1 to remove the difference
    utt_id = spk_id + "_{0:03}".format(num)
    return utt_id


def transform_the_input_text(_type, input_text):

    # There may be some space in the text due to misreading, remove them.
    input_text = input_text.strip().replace("　", "")
    if _type == "original":
        return input_text + "."

    output_text = ""

    if _type == "pinyin":
        style = Style.TONE3
        pinyin_list = lazy_pinyin(input_text, style=style)
        for s in pinyin_list:
            output_text += s + " "
        return output_text
    else:
        style = Style.BOPOMOFO
        pinyin_list = lazy_pinyin(input_text, style=style)
        if _type == "phoneme":
            syllable_map = pt.SYLLABLE_MAP(source="Zhuyin", target="IPA")
            for s in pinyin_list:
                tone = pt.get_chewing_tone(s)
                syllable = re.sub("[ˊˇˋ˙]", "", s)

                # NOTE(unilight): some BOPOMOFO do not exist in IPA table, causing some errors. We just ignore them
                try:
                    ipa = syllable_map.transform(syllable).format(tone)
                    output_text += ipa + " "
                except Exception:
                    pass
        elif _type == "zhuyin":
            for s in pinyin_list:
                output_text += s + " "
        else:
            raise Exception(
                "Unknown transform type {}. The argument should be original, pinyin, phoneme or zhuyin.".format(
                    _type
                )
            )
    output_text = output_text[:-1] + "."
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--spk_id", type=str, help="the current speaker id")
    parser.add_argument(
        "-f", "--file_name", type=str, help="the text file that need to be handled"
    )
    parser.add_argument(
        "-t",
        "--transform_type",
        type=str,
        help="the char type that we thansform the chinese text into, could be original (nothing changed), phoneme, roman pinyin, and zhuyin",
        default="phoneme",
    )

    args = parser.parse_args()

    with open(args.file_name, "r", encoding="utf-8") as text_file:
        for num, input_text in enumerate(text_file.readlines()):
            utt_id = get_the_utt_id(num, args.spk_id)
            output_text = transform_the_input_text(args.transform_type, input_text)
            print(utt_id + " " + output_text)
