"""
Decode model
"""
import argparse

import torch

from dataloader import BOS, EOS, UNK_IDX
from decoding import Decoder
from model import dummy_mask
from util import maybe_mkdir, unpack_batch


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', required=True, help='Dev/Test file')
    parser.add_argument('--out_file', required=True, help='Output file')
    parser.add_argument('--lang', required=True, help='Language tag')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--decode', default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_size', default=5, type=int)
    return parser.parse_args()
    # fmt: on


def read_file(filename, lang_tag):
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp.readlines():
            lemma, _, tags = line.strip().split("\t")
            yield list(lemma), [lang_tag] + tags.split(";")


def encode(model, lemma, tags, device):
    tag_shift = model.src_vocab_size - len(model.attr_c2i)

    src = []
    src.append(model.src_c2i[BOS])
    for char in lemma:
        src.append(model.src_c2i.get(char, UNK_IDX))
    src.append(model.src_c2i[EOS])

    attr = [0] * (len(model.attr_c2i) + 1)
    for tag in tags:
        if tag in model.attr_c2i:
            attr_idx = model.attr_c2i[tag] - tag_shift
        else:
            attr_idx = -1
        if attr[attr_idx] == 0:
            attr[attr_idx] = model.attr_c2i.get(tag, UNK_IDX)

    return (
        torch.tensor(src, device=device).view(len(src), 1),
        torch.tensor(attr, device=device).view(1, len(attr)),
    )


def main():
    opt = get_args()

    decode_fn = Decoder(opt.decode, max_len=opt.max_len, beam_size=opt.beam_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(open(opt.model, mode="rb"), map_location=device)
    model = model.to(device)

    trg_i2c = {i: c for c, i in model.trg_c2i.items()}

    def decode_trg(seq):
        return [trg_i2c[i] for i in seq]

    maybe_mkdir(opt.out_file)
    with open(opt.out_file, "w", encoding="utf-8") as fp:
        for lemma, tags in read_file(opt.in_file, opt.lang):
            src = encode(model, lemma, tags, device)
            src_mask = dummy_mask(src)
            pred, _ = decode_fn(model, src, src_mask)
            pred = unpack_batch(pred)[0]
            pred_out = "".join(decode_trg(pred))
            fp.write(f'{"".join(lemma)}\t{pred_out}\t{";".join(tags[1:])}\n')


if __name__ == "__main__":
    with torch.no_grad():
        main()
