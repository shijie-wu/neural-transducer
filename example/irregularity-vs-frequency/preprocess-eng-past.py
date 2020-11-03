import argparse
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def maybe_mkdir(filename):
    """
    maybe mkdir
    """
    path = os.path.dirname(filename)
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimorph", required=True, type=str)
    parser.add_argument("--past", required=True, type=str)
    parser.add_argument("--outdir")
    parser.add_argument("--split", default=0.2, type=float)
    parser.add_argument("--nchar", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--mode", required=True, choices=["odonnell", "albright"])
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--suffix", action="store_true")
    return parser.parse_args()


def edit_distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    action = dict()
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
        action[(i, 0)] = ((i - 1, 0), f"I({str2[i-1]})")
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
        action[(0, j)] = ((0, j - 1), f"D({str1[j-1]})")
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(
                table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + dg
            )
            if table[i][j] == table[i - 1][j] + 1:
                action[(i, j)] = ((i - 1, j), f"I({str2[i-1]})")
            elif table[i][j] == table[i][j - 1] + 1:
                action[(i, j)] = ((i, j - 1), f"D({str1[j-1]})")
            elif table[i][j] == table[i - 1][j - 1] + dg:
                if dg == 1:
                    act = f"R({str1[j-1]},{str2[i-1]})"
                else:
                    act = f"C({str1[j-1]})"
                action[(i, j)] = ((i - 1, j - 1), act)
            else:
                raise ValueError
    edit_script = []
    back_ptr = (len(str2), len(str1))
    while back_ptr != (0, 0):
        back_ptr, edit = action[back_ptr]
        edit_script.append(edit)
    # return int(table[len(str2)][len(str1)])
    return edit_script[::-1]


def match_script(short_script, long_script):
    i, j = 0, 0
    match = True
    while i < len(short_script) and j < len(long_script):
        if short_script[i] == long_script[j]:
            i += 1
            j += 1
        elif long_script[j][0] == "C":
            j += 1
        else:
            return not match
    if i == len(short_script) and j == len(long_script):
        return match
    else:
        return not match


class Preprocesser(object):
    def __init__(self, opt):
        self.opt = opt

    def read_unimorph_data(self, file):
        raise NotImplementedError

    def read_data(self, file):
        raise NotImplementedError

    def match_edit_script(self, short_script, long_script):
        raise NotImplementedError

    def find_substr(self, unimorph_ed, past_ed):
        opt = self.opt
        # Step 1. Remove dervation from unimorph
        clean_lemma = list()
        for word in unimorph_ed:
            if len(word) < opt.nchar:
                clean_lemma.append(word)
                continue
            prefix_lst = []
            suffix_lst = []
            for i in range(opt.nchar, len(word)):
                prefix, suffix = word[:i], word[i:]
                if opt.prefix and len(prefix) >= opt.nchar:
                    if prefix in unimorph_ed and self.match_edit_script(
                        unimorph_ed[prefix], unimorph_ed[word]
                    ):
                        prefix_lst.append(prefix)
                if opt.suffix and len(suffix) >= opt.nchar:
                    if suffix in unimorph_ed and self.match_edit_script(
                        unimorph_ed[suffix], unimorph_ed[word]
                    ):
                        suffix_lst.append(suffix)
            if prefix_lst or suffix_lst:
                pass
            else:
                clean_lemma.append(word)

        # Step 2. Remove lemma which dervation is in past
        clean_lemma_set = set(clean_lemma)
        for word in past_ed:
            if len(word) < opt.nchar:
                clean_lemma.append(word)
                continue
            for i in range(opt.nchar, len(word)):
                prefix, suffix = word[:i], word[i:]
                if opt.prefix and len(prefix) >= opt.nchar:
                    if prefix in clean_lemma_set and self.match_edit_script(
                        unimorph_ed[prefix], past_ed[word]
                    ):
                        clean_lemma_set.remove(prefix)
                if opt.suffix and len(suffix) >= opt.nchar:
                    if suffix in clean_lemma_set and self.match_edit_script(
                        unimorph_ed[suffix], past_ed[word]
                    ):
                        clean_lemma_set.remove(suffix)

        # Step 3. Remove duplcation
        final_lemma = []
        for lemma in clean_lemma_set:
            if lemma not in past_ed:
                final_lemma.append(lemma)

        return final_lemma

    def run(self):
        opt = self.opt
        unimorph_data, unimorph_edit_script = self.read_unimorph_data(opt.unimorph)
        past_data, past_edit_script = self.read_data(opt.past)
        lemma = self.find_substr(unimorph_edit_script, past_edit_script)

        if opt.outdir:
            maybe_mkdir(opt.outdir)
            lemma = sorted(lemma)
            lemma = np.random.RandomState(opt.seed).permutation(lemma)
            split = int(opt.split * len(lemma))
            dev = lemma[:split]
            train = lemma[split:]
            test = list(past_data.keys())
            with open(f"{opt.outdir}.train", "w", encoding="utf-8") as fp:
                for lemma in train:
                    fp.writelines(unimorph_data[lemma])
            with open(f"{opt.outdir}.dev", "w", encoding="utf-8") as fp:
                for lemma in dev:
                    fp.writelines(unimorph_data[lemma])
            with open(f"{opt.outdir}.test", "w", encoding="utf-8") as fp:
                for lemma in test:
                    fp.writelines(past_data[lemma])


class ODonnellPreprocesser(Preprocesser):
    def __init__(self, opt):
        super().__init__(opt)
        self.TAG = {
            "VBZ": "V;3;SG;PRS",
            "VB": "V;NFIN",
            "VBP": "V;NFIN",
            "VBD": "V;PST",
            "VBG": "V;V.PTCP;PRS",
            "VBN": "V;V.PTCP;PST",
        }

    def read_unimorph_data(self, file):
        data = defaultdict(list)
        edit = defaultdict(dict)
        with open(file, "r", encoding="utf-8") as fp:
            for line in tqdm(fp.readlines()):
                if line == "\n":
                    continue
                lemma, word, tags = line.strip().split("\t")
                edit_script = edit_distance(lemma, word)
                data[lemma].append(line)
                edit[lemma][tags] = edit_script
        return data, edit

    def read_data(self, file):
        data = defaultdict(list)
        edit = defaultdict(dict)
        with open(file, "r", encoding="utf-8") as fp:
            _ = fp.readline()
            for line in fp.readlines():
                if line == "\n":
                    continue
                toks = line.strip().split(",")
                lemma = toks[6]
                word = toks[4]
                tags = self.TAG[toks[7]]
                edit_script = edit_distance(lemma, word)
                data[lemma].append("\t".join([lemma, word, tags]) + "\n")
                edit[lemma][tags] = edit_script
        return data, edit

    def match_edit_script(self, short_script, long_script):
        subword_tags = set(short_script.keys())
        word_tags = set(long_script.keys())
        match = True
        for tags in subword_tags.intersection(word_tags):
            if not match_script(short_script[tags], long_script[tags]):
                return not match
        return match


class AlbrightPreprocesser(Preprocesser):
    def __init__(self, opt):
        super().__init__(opt)
        self.TAG = "V;PST"

    def read_unimorph_data(self, file):
        data, edit = dict(), dict()
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                if line == "\n":
                    continue
                lemma, word, tags = line.strip().split("\t")
                if tags == self.TAG:
                    edit_script = edit_distance(lemma, word)
                    data[lemma] = "\t".join([lemma, word, tags]) + "\n"
                    edit[lemma] = edit_script
        return data, edit

    def read_data(self, file):
        data, edit = dict(), dict()
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                if line == "\n":
                    continue
                lemma, word, *_ = line.strip().split("\t")
                edit_script = edit_distance(lemma, word)
                data[lemma] = "\t".join([lemma, word, self.TAG]) + "\n"
                edit[lemma] = edit_script
        return data, edit

    def match_edit_script(self, short_script, long_script):
        return match_script(short_script, long_script)


def main():
    opt = get_args()
    if opt.mode == "odonnell":
        runner = ODonnellPreprocesser(opt)
    elif opt.mode == "albright":
        runner = AlbrightPreprocesser(opt)
    else:
        raise ValueError
    runner.run()


if __name__ == "__main__":
    main()
