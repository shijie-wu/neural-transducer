import os

import fire

MAX_HALL = 10_000


def read_file(file):
    with open(file) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            toks = line.split("\t")
            if len(toks) == 3:
                yield toks


def halluication(directory, lang):
    mode = "train"
    if not os.path.isfile(f"{directory}/{lang}.hall"):
        print("missing .hall for", lang)
        return
    with open(f"{directory}/{lang}.hall.{mode}", "w") as fp:
        for toks in read_file(f"{directory}/{lang}.{mode}"):
            print(*toks, sep="\t", file=fp)
        for i, toks in enumerate(read_file(f"{directory}/{lang}.hall")):
            if i == MAX_HALL:
                break
            print(toks[0], toks[1], f"fake;{toks[2]}", sep="\t", file=fp)


if __name__ == "__main__":
    fire.Fire(halluication)
