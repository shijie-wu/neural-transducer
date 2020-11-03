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
    parser.add_argument("--infile", required=True, type=str)
    parser.add_argument("--outdir")
    parser.add_argument("--nfold", default=10, type=int)
    parser.add_argument("--nchar", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
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


def read_data(file):
    data = defaultdict(list)
    data_edit = defaultdict(dict)
    with open(file, "r", encoding="utf-8") as fp:
        for line in tqdm(fp.readlines()):
            if line == "\n":
                continue
            lemma, word, tags = line.strip().split("\t")
            edit_script = edit_distance(lemma, word)
            data[lemma].append(line)
            data_edit[lemma][tags] = edit_script
    return data, data_edit


def write_data(file, data, train, dev, test):
    for fp, group in [
        (f"{file}.train", train),
        (f"{file}.dev", dev),
        (f"{file}.test", test),
    ]:
        with open(fp, "w", encoding="utf-8") as fp:
            for lemma in group:
                fp.writelines(data[lemma])


def match_with_edit_script(edit_script, subword, word):
    def match_script(short_script, long_script):
        i, j = 0, 0
        while i < len(short_script) and j < len(long_script):
            if short_script[i] == long_script[j]:
                i += 1
                j += 1
            elif long_script[j][0] == "C":
                j += 1
            else:
                return False
        if i == len(short_script) and j == len(long_script):
            return True
        else:
            return False

    subword_tags = set(edit_script[subword].keys())
    word_tags = set(edit_script[word].keys())
    match = True
    for tags in subword_tags.intersection(word_tags):
        if not match_script(edit_script[subword][tags], edit_script[word][tags]):
            return not match
    return match


def find_substr(
    lemma, edit_script, min_word=1, log=False, cnt_prefix=False, cnt_suffix=False
):
    cnt = 0
    clean_lemma = []
    for word in lemma:
        if len(word) < min_word:
            clean_lemma.append(word)
            continue
        prefix_lst = []
        suffix_lst = []
        for i in range(min_word, len(word)):
            prefix, suffix = word[:i], word[i:]
            if cnt_prefix and len(prefix) >= min_word and prefix in lemma:
                if match_with_edit_script(edit_script, prefix, word):
                    prefix_lst.append(prefix)
            if cnt_suffix and len(suffix) >= min_word and suffix in lemma:
                if match_with_edit_script(edit_script, suffix, word):
                    suffix_lst.append(suffix)
        if prefix_lst or suffix_lst:
            cnt += 1
            if prefix_lst and log:
                print(f"prefix {prefix_lst} -> {word}")
            if suffix_lst and log:
                print(f"suffix {suffix_lst} -> {word}")
        else:
            clean_lemma.append(word)
    print(f"{min_word},{cnt},{round(cnt/len(lemma)*100, 2)}")
    # print(f'|{min_word}|{cnt}|{round(cnt/len(lemma)*100, 2)}|')
    return clean_lemma


def main():
    opt = get_args()
    assert os.path.isfile(opt.infile)
    data, edit_script = read_data(opt.infile)
    lemma = set(data.keys())
    clean_lemma = find_substr(
        lemma, edit_script, opt.nchar, opt.log, opt.prefix, opt.suffix
    )
    if opt.outdir:
        maybe_mkdir(opt.outdir)
        clean_lemma = sorted(clean_lemma)
        clean_lemma = np.random.RandomState(opt.seed).permutation(clean_lemma)
        lemma_folds = np.array_split(clean_lemma, opt.nfold)
        for i in range(opt.nfold):
            ids = list(range(opt.nfold))
            dev = ids[i]
            test = ids[(i + 1) % opt.nfold]
            ids.remove(dev)
            ids.remove(test)
            train = np.concatenate([lemma_folds[idx] for idx in ids])
            dev = lemma_folds[dev]
            test = lemma_folds[test]
            write_data(f"{opt.outdir}.{i + 1}", data, train, dev, test)


if __name__ == "__main__":
    main()
