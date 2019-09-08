'''
Decode model
'''
import argparse
from functools import partial

import torch

from dataloader import BOS, EOS, UNK_IDX
from decoding import decode_beam_search, decode_greedy
from util import maybe_mkdir


def get_args():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', required=True, help='Dev/Test file')
    parser.add_argument('--out_file', required=True, help='Output file')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--decode', default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--nonorm', default=False, action='store_true')
    return parser.parse_args()
    # yapf: enable


def setup_inference(opt):
    decode_fn = None
    if opt.decode == 'greedy':
        decode_fn = partial(decode_greedy, max_len=opt.max_len)
    elif opt.decode == 'beam':
        decode_fn = partial(
            decode_beam_search,
            max_len=opt.max_len,
            nb_beam=opt.beam_size,
            norm=not opt.nonorm)
    return decode_fn


def encode(model, word, tags, device):
    tag_shift = model.src_vocab_size - len(model.attr_c2i)

    src = []
    src.append(model.src_c2i[BOS])
    for char in word:
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

    return (torch.tensor(src, device=device).view(len(src), 1),
            torch.tensor(attr, device=device).view(1, len(attr)))


def main():
    opt = get_args()

    decode_fn = setup_inference(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(open(opt.model, mode='rb'), map_location=device)
    model = model.to(device)

    trg_i2c = {i: c for c, i in model.trg_c2i.items()}
    decode_trg = lambda seq: [trg_i2c[i] for i in seq]

    maybe_mkdir(opt.out_file)
    with open(opt.in_file, 'r', encoding='utf-8') as in_fp, \
         open(opt.out_file, 'w', encoding='utf-8') as out_fp:
        for line in in_fp.readlines():
            toks = line.strip().split('\t')
            if len(toks) < 2 or line[0] == '#':  # pass through
                out_fp.write(line)
                continue
            # word, lemma, tags = toks[1], toks[2], toks[5]
            word, tags = toks[1], toks[5]
            word, tags = list(word), tags.split(';')
            src = encode(model, word, tags, device)
            pred, _ = decode_fn(model, src)
            pred_out = ''.join(decode_trg(pred))
            # write lemma
            toks[2] = pred_out
            out_fp.write('\t'.join(toks) + '\n')


if __name__ == '__main__':
    with torch.no_grad():
        main()
