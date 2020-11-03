import argparse
import json
import os
from collections import Counter
from functools import partial

import langcodes
from nltk.tokenize import word_tokenize
from smart_open import smart_open
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testfiles", nargs="*", required=True)
    parser.add_argument("--wiki", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--outfile", required=True)
    return parser.parse_args()


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


def read_forms(files, ignore_case=True):
    forms = set()
    for file in files:
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                if not line.strip():
                    continue
                word = line.split("\t")[1]
                if ignore_case:
                    word = word.lower()
                forms.add(word)
    return forms


def main():
    opt = get_args()
    forms = read_forms(opt.testfiles)
    cnt = Counter()
    max_space = max([len(w.split()) for w in forms])
    language = langcodes.Language.get(opt.lang).language_name().lower()

    print(f"max space = {max_space}")
    # print(language, max_space, vocab_punct)

    tokenize = partial(word_tokenize, language=language)

    nb_article = 0
    for line in tqdm(smart_open(opt.wiki)):
        nb_article += 1
    for line in tqdm(smart_open(opt.wiki), total=nb_article):
        article = json.loads(line)
        text = " ".join(article["section_texts"]).lower()
        try:
            tokens = tokenize(text)
        except LookupError:
            print("Using default tokenizer")
            tokenize = partial(word_tokenize, language="english")
            tokens = tokenize(text)
        for span in range(max_space):
            for i in range(len(tokens) - span):
                word = " ".join(tokens[i : i + span + 1])
                if word in forms:
                    cnt[word] += 1
    maybe_mkdir(opt.outfile)
    with open(opt.outfile, "w", encoding="utf-8") as fp:
        for word in sorted(forms):
            wordcnt = cnt[word]
            fp.write(f"{word}\t{wordcnt}\n")


if __name__ == "__main__":
    main()
