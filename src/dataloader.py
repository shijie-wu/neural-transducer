import xml.etree.ElementTree
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from align import Aligner

BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"
ALIGN = "<a>"
STEP = "<step>"
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
STEP_IDX = 4


class Dataloader(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2SeqDataLoader(Dataloader):
    def __init__(
        self,
        train_file: List[str],
        dev_file: List[str],
        test_file: Optional[List[str]] = None,
        shuffle=False,
    ):
        super().__init__()
        self.train_file = train_file[0] if len(train_file) == 1 else train_file
        self.dev_file = dev_file[0] if len(dev_file) == 1 else dev_file
        self.test_file = (
            test_file[0] if test_file and len(test_file) == 1 else test_file
        )
        self.shuffle = shuffle
        self.batch_data: Dict[str, List] = dict()
        self.nb_train, self.nb_dev, self.nb_test = 0, 0, 0
        self.nb_attr = 0
        self.source, self.target = self.build_vocab()
        self.source_vocab_size = len(self.source)
        self.target_vocab_size = len(self.target)
        self.attr_c2i: Optional[Dict]
        if self.nb_attr > 0:
            self.source_c2i = {c: i for i, c in enumerate(self.source[: -self.nb_attr])}
            self.attr_c2i = {
                c: i + len(self.source_c2i)
                for i, c in enumerate(self.source[-self.nb_attr :])
            }
        else:
            self.source_c2i = {c: i for i, c in enumerate(self.source)}
            self.attr_c2i = None
        self.target_c2i = {c: i for i, c in enumerate(self.target)}
        self.sanity_check()

    def sanity_check(self):
        assert self.source[PAD_IDX] == PAD
        assert self.target[PAD_IDX] == PAD
        assert self.source[BOS_IDX] == BOS
        assert self.target[BOS_IDX] == BOS
        assert self.source[EOS_IDX] == EOS
        assert self.target[EOS_IDX] == EOS
        assert self.source[UNK_IDX] == UNK
        assert self.target[UNK_IDX] == UNK

    def build_vocab(self):
        src_set, trg_set = set(), set()
        self.nb_train = 0
        for src, trg in self.read_file(self.train_file):
            self.nb_train += 1
            src_set.update(src)
            trg_set.update(trg)
        self.nb_dev = sum([1 for _ in self.read_file(self.dev_file)])
        if self.test_file is not None:
            self.nb_test = sum([1 for _ in self.read_file(self.test_file)])
        source = [PAD, BOS, EOS, UNK] + sorted(list(src_set))
        target = [PAD, BOS, EOS, UNK] + sorted(list(trg_set))
        return source, target

    def read_file(self, file):
        raise NotImplementedError

    def _file_identifier(self, file):
        return file

    def list_to_tensor(self, lst: List[List[int]], max_seq_len=None):
        max_len = max([len(x) for x in lst])
        if max_seq_len is not None:
            max_len = min(max_len, max_seq_len)
        data = torch.zeros((max_len, len(lst)), dtype=torch.long)
        for i, seq in tqdm(enumerate(lst), desc="build tensor"):
            data[: len(seq), i] = torch.tensor(seq)
        mask = (data > 0).float()
        return data, mask

    def _batch_sample(self, batch_size, file, shuffle):
        key = self._file_identifier(file)
        if key not in self.batch_data:
            lst = list()
            for src, trg in tqdm(self._iter_helper(file), desc="read file"):
                lst.append((src, trg))
            src_data, src_mask = self.list_to_tensor([src for src, _ in lst])
            trg_data, trg_mask = self.list_to_tensor([trg for _, trg in lst])
            self.batch_data[key] = (src_data, src_mask, trg_data, trg_mask)

        src_data, src_mask, trg_data, trg_mask = self.batch_data[key]
        nb_example = len(src_data[0])
        if shuffle:
            idx = np.random.permutation(nb_example)
        else:
            idx = np.arange(nb_example)
        for start in range(0, nb_example, batch_size):
            idx_ = idx[start : start + batch_size]
            src_mask_b = src_mask[:, idx_]
            trg_mask_b = trg_mask[:, idx_]
            src_len = int(src_mask_b.sum(dim=0).max().item())
            trg_len = int(trg_mask_b.sum(dim=0).max().item())
            src_data_b = src_data[:src_len, idx_].to(self.device)
            trg_data_b = trg_data[:trg_len, idx_].to(self.device)
            src_mask_b = src_mask_b[:src_len].to(self.device)
            trg_mask_b = trg_mask_b[:trg_len].to(self.device)
            yield (src_data_b, src_mask_b, trg_data_b, trg_mask_b)

    def train_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.train_file, shuffle=self.shuffle)

    def dev_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.dev_file, shuffle=False)

    def test_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.test_file, shuffle=False)

    def encode_source(self, sent):
        if sent[0] != BOS:
            sent = [BOS] + sent
        if sent[-1] != EOS:
            sent = sent + [EOS]
        seq_len = len(sent)
        s = []
        for x in sent:
            if x in self.source_c2i:
                s.append(self.source_c2i[x])
            else:
                s.append(self.attr_c2i[x])
        return torch.tensor(s, device=self.device).view(seq_len, 1)

    def decode_source(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.source[x] for x in sent]

    def decode_target(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.target[x] for x in sent]

    def _sample(self, file):
        for src, trg in self._iter_helper(file):
            yield (
                torch.tensor(src, device=self.device).view(len(src), 1),
                torch.tensor(trg, device=self.device).view(len(trg), 1),
            )

    def train_sample(self):
        yield from self._sample(self.train_file)

    def dev_sample(self):
        yield from self._sample(self.dev_file)

    def test_sample(self):
        yield from self._sample(self.test_file)

    def _iter_helper(self, file):
        for source, target in self.read_file(file):
            src = [self.source_c2i[BOS]]
            for s in source:
                src.append(self.source_c2i.get(s, UNK_IDX))
            src.append(self.source_c2i[EOS])
            trg = [self.target_c2i[BOS]]
            for t in target:
                trg.append(self.target_c2i.get(t, UNK_IDX))
            trg.append(self.target_c2i[EOS])
            yield src, trg


class AlignSeq2SeqDataLoader(Seq2SeqDataLoader):
    def __init__(
        self,
        train_file: List[str],
        dev_file: List[str],
        test_file: Optional[List[str]] = None,
        shuffle=False,
    ):
        self.data: Dict[str, List] = dict()
        super().__init__(train_file, dev_file, test_file, shuffle)

    def sanity_check(self):
        super().sanity_check()
        assert self.target[STEP_IDX] == STEP

    def gen_act(self, src, trg):
        assert len(src) == len(trg)
        s = []
        for idx in range(len(src)):
            if trg[idx] == ALIGN:
                s.append(STEP)
            else:
                s.append(trg[idx])
                if idx + 1 < len(src) and src[idx + 1] != ALIGN and trg[idx] != EOS:
                    s.append(STEP)
        return s

    def read_file(self, file):
        if file not in self.data:
            pair = []
            data = []
            for item in super().read_file(file):
                assert len(item) >= 2
                src, trg, rest = item[0], item[1], item[2:]
                pair.append(([BOS] + src + [EOS], [BOS] + trg + [EOS]))
                data.append(rest)
            align = Aligner(pair, align_symbol=ALIGN)
            assert len(pair) == len(data) == len(align.alignedpairs)
            for idx in range(len(pair)):
                action = self.gen_act(*align.alignedpairs[idx])
                step_cnt = sum([int(x == STEP) for x in action])
                assert step_cnt + 1 == len(
                    pair[idx][0]
                ), "step cnt {}\n{}\n{}\n{}".format(
                    step_cnt, pair[idx], action, align.alignedpairs[idx]
                )
                data[idx] = tuple([pair[idx][0], action, *data[idx]])
            self.data[file] = data
        yield from self.data[file]

    def _iter_helper(self, file):
        for source, target in self.read_file(file):
            src = []
            for s in source:
                src.append(self.source_c2i.get(s, UNK_IDX))
            trg = []
            for t in target:
                trg.append(self.target_c2i.get(t, UNK_IDX))
            yield src, trg

    def build_vocab(self):
        source, target = super().build_vocab()
        source = [x for x in source if x not in set([PAD, BOS, EOS, UNK, STEP])]
        target = [x for x in target if x not in set([PAD, BOS, EOS, UNK, STEP])]
        source = [PAD, BOS, EOS, UNK] + source
        target = [PAD, BOS, EOS, UNK, STEP] + target
        return source, target


class SIGMORPHON2017Task1(Seq2SeqDataLoader):
    def build_vocab(self):
        char_set, tag_set = set(), set()
        self.nb_train = 0
        for lemma, word, tags in self.read_file(self.train_file):
            self.nb_train += 1
            char_set.update(lemma)
            char_set.update(word)
            tag_set.update(tags)
        self.nb_dev = sum([1 for _ in self.read_file(self.dev_file)])
        if self.test_file is not None:
            self.nb_test = sum([1 for _ in self.read_file(self.test_file)])
        chars = sorted(list(char_set))
        tags = sorted(list(tag_set))
        self.nb_attr = len(tags)
        source = [PAD, BOS, EOS, UNK] + chars + tags
        target = [PAD, BOS, EOS, UNK] + chars
        return source, target

    def read_file(self, file):
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                line = line.strip()
                if not line:
                    continue
                toks = line.split("\t")
                if len(toks) != 3:
                    print("WARNING: missing tokens", toks)
                    continue
                lemma, word, tags = toks
                yield list(lemma), list(word), tags.split(";")

    def _iter_helper(self, file):
        for lemma, word, tags in self.read_file(file):
            src = [self.source_c2i[BOS]]
            for tag in tags:
                src.append(self.attr_c2i.get(tag, UNK_IDX))
            for char in lemma:
                src.append(self.source_c2i.get(char, UNK_IDX))
            src.append(self.source_c2i[EOS])
            trg = [self.target_c2i[BOS]]
            for char in word:
                trg.append(self.target_c2i.get(char, UNK_IDX))
            trg.append(self.target_c2i[EOS])
            yield src, trg


class Unimorph(SIGMORPHON2017Task1):
    def build_vocab(self):
        char_set, tag_set = set(), set()
        cnts = []
        for fp in [self.train_file, self.dev_file, self.test_file]:
            cnt = 0
            for lemma, word, tags in self.read_file(fp):
                cnt += 1
                char_set.update(lemma)
                char_set.update(word)
                tag_set.update(tags)
            cnts.append(cnt)
        self.nb_train, self.nb_dev, self.nb_test = cnts
        chars = sorted(list(char_set))
        tags = sorted(list(tag_set))
        self.nb_attr = len(tags)
        source = [PAD, BOS, EOS, UNK] + chars + tags
        target = [PAD, BOS, EOS, UNK] + chars
        return source, target


class SIGMORPHON2016Task1(SIGMORPHON2017Task1):
    def read_file(self, file):
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                lemma, tags, word = line.strip().split("\t")
                yield list(lemma), list(word), tags.split(",")


class Lemmatization(SIGMORPHON2017Task1):
    def read_file(self, file):
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                word, lemma, tags = line.strip().split("\t")
                yield list(word.lower()), list(lemma.lower()), tags.split("|")


class LemmatizationNotag(Seq2SeqDataLoader):
    def read_file(self, file):
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                word, lemma, tags = line.strip().split("\t")
                yield list(word.lower()), list(lemma.lower())


class AlignSIGMORPHON2017Task1(AlignSeq2SeqDataLoader, SIGMORPHON2017Task1):
    def _iter_helper(self, file):
        tag_shift = len(self.source) - self.nb_attr
        for lemma, word, tags in self.read_file(file):
            src = []
            for char in lemma:
                src.append(self.source_c2i.get(char, UNK_IDX))
            trg = []
            for char in word:
                trg.append(self.target_c2i.get(char, UNK_IDX))
            attr = [0] * (self.nb_attr + 1)
            for tag in tags:
                if tag in self.attr_c2i:
                    attr_idx = self.attr_c2i[tag] - tag_shift
                else:
                    attr_idx = -1
                if attr[attr_idx] == 0:
                    attr[attr_idx] = self.attr_c2i.get(tag, UNK_IDX)
            yield src, trg, attr

    def _batch_sample(self, batch_size, file, shuffle):
        key = self._file_identifier(file)
        if key not in self.batch_data:
            lst = list()
            for src, trg, attr in tqdm(self._iter_helper(file), desc="read file"):
                lst.append((src, trg, attr))
            src_data, src_mask = self.list_to_tensor([src for src, _, _ in lst])
            trg_data, trg_mask = self.list_to_tensor([trg for _, trg, _ in lst])
            attr_data, _ = self.list_to_tensor([attr for _, _, attr in lst])
            attr_data = attr_data.transpose(0, 1)
            data = ((src_data, attr_data), src_mask, trg_data, trg_mask)
            self.batch_data[key] = data

        data = self.batch_data[key]
        (src_data, attr_data), src_mask, trg_data, trg_mask = data
        nb_example = len(src_data[0])
        if shuffle:
            idx = np.random.permutation(nb_example)
        else:
            idx = np.arange(nb_example)
        for start in range(0, nb_example, batch_size):
            idx_ = idx[start : start + batch_size]
            src_mask_b = src_mask[:, idx_]
            trg_mask_b = trg_mask[:, idx_]
            src_len = int(src_mask_b.sum(dim=0).max().item())
            trg_len = int(trg_mask_b.sum(dim=0).max().item())
            src_data_b = src_data[:src_len, idx_].to(self.device)
            trg_data_b = trg_data[:trg_len, idx_].to(self.device)
            src_mask_b = src_mask_b[:src_len].to(self.device)
            trg_mask_b = trg_mask_b[:trg_len].to(self.device)
            attr_data_b = attr_data[idx_, :].to(self.device)
            yield ((src_data_b, attr_data_b), src_mask_b, trg_data_b, trg_mask_b)

    def _sample(self, file):
        for src, trg, tags in self._iter_helper(file):
            yield (
                (
                    torch.tensor(src, device=self.device).view(len(src), 1),
                    torch.tensor(tags, device=self.device).view(1, len(tags)),
                ),
                torch.tensor(trg, device=self.device).view(len(trg), 1),
            )


class TagSIGMORPHON2017Task1(SIGMORPHON2017Task1):
    def _iter_helper(self, file):
        tag_shift = len(self.source) - self.nb_attr
        for lemma, word, tags in self.read_file(file):
            src = []
            src.append(self.source_c2i[BOS])
            for char in lemma:
                src.append(self.source_c2i.get(char, UNK_IDX))
            src.append(self.source_c2i[EOS])
            trg = []
            trg.append(self.target_c2i[BOS])
            for char in word:
                trg.append(self.target_c2i.get(char, UNK_IDX))
            trg.append(self.target_c2i[EOS])
            attr = [0] * (self.nb_attr + 1)
            for tag in tags:
                if tag in self.attr_c2i:
                    attr_idx = self.attr_c2i[tag] - tag_shift
                else:
                    attr_idx = -1
                if attr[attr_idx] == 0:
                    attr[attr_idx] = self.attr_c2i.get(tag, UNK_IDX)
            yield src, trg, attr

    def _batch_sample(self, batch_size, file, shuffle):
        key = self._file_identifier(file)
        if key not in self.batch_data:
            lst = list()
            for src, trg, attr in tqdm(self._iter_helper(file), desc="read file"):
                lst.append((src, trg, attr))
            src_data, src_mask = self.list_to_tensor([src for src, _, _ in lst])
            trg_data, trg_mask = self.list_to_tensor([trg for _, trg, _ in lst])
            attr_data, _ = self.list_to_tensor([attr for _, _, attr in lst])
            attr_data = attr_data.transpose(0, 1)
            data = ((src_data, attr_data), src_mask, trg_data, trg_mask)
            self.batch_data[key] = data

        data = self.batch_data[key]
        (src_data, attr_data), src_mask, trg_data, trg_mask = data
        nb_example = len(src_data[0])
        if shuffle:
            idx = np.random.permutation(nb_example)
        else:
            idx = np.arange(nb_example)
        for start in range(0, nb_example, batch_size):
            idx_ = idx[start : start + batch_size]
            src_mask_b = src_mask[:, idx_]
            trg_mask_b = trg_mask[:, idx_]
            src_len = int(src_mask_b.sum(dim=0).max().item())
            trg_len = int(trg_mask_b.sum(dim=0).max().item())
            src_data_b = src_data[:src_len, idx_].to(self.device)
            trg_data_b = trg_data[:trg_len, idx_].to(self.device)
            src_mask_b = src_mask_b[:src_len].to(self.device)
            trg_mask_b = trg_mask_b[:trg_len].to(self.device)
            attr_data_b = attr_data[idx_, :].to(self.device)
            yield ((src_data_b, attr_data_b), src_mask_b, trg_data_b, trg_mask_b)

    def _sample(self, file):
        for src, trg, tags in self._iter_helper(file):
            yield (
                (
                    torch.tensor(src, device=self.device).view(len(src), 1),
                    torch.tensor(tags, device=self.device).view(1, len(tags)),
                ),
                torch.tensor(trg, device=self.device).view(len(trg), 1),
            )


class TagSIGMORPHON2016Task1(SIGMORPHON2016Task1, TagSIGMORPHON2017Task1):
    pass


class TagUnimorph(Unimorph, TagSIGMORPHON2017Task1):
    pass


class TagSIGMORPHON2019Task1(TagSIGMORPHON2017Task1):
    def read_file(self, file):
        if "train" in file:
            lang_tag = [file.split("/")[-1].split("-train")[0]]
        elif "dev" in file:
            lang_tag = [file.split("/")[-1].split("-dev")[0]]
        else:
            raise ValueError
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                lemma, word, tags = line.strip().split("\t")
                yield list(lemma), list(word), tags.split(";") + lang_tag

    def _iter_helper(self, file):
        if not isinstance(file, list):
            file = [file]
        for fp in file:
            yield from super()._iter_helper(fp)

    def _file_identifier(self, file):
        if isinstance(file, list):
            return tuple(sorted(file))
        else:
            return file

    def build_vocab(self):
        char_set, tag_set = set(), set()
        cnts = []
        for fp in self.train_file:
            cnt = 0
            for lemma, word, tags in self.read_file(fp):
                cnt += 1
                char_set.update(lemma)
                char_set.update(word)
                tag_set.update(tags)
            cnts.append(cnt)
        self.nb_train = cnts[0]
        self.nb_dev = sum([1 for _ in self.read_file(self.dev_file)])
        if self.test_file is None:
            self.nb_test = 0
        else:
            self.nb_test = sum([1 for _ in self.read_file(self.test_file)])
        chars = sorted(list(char_set))
        tags = sorted(list(tag_set))
        self.nb_attr = len(tags)
        source = [PAD, BOS, EOS, UNK] + chars + tags
        target = [PAD, BOS, EOS, UNK] + chars
        return source, target


class TagSIGMORPHON2019Task2(TagSIGMORPHON2019Task1):
    def read_file(self, file):
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                toks = line.strip().split("\t")
                if len(toks) < 2 or line[0] == "#":
                    continue
                word, lemma, tags = toks[1], toks[2], toks[5]
                yield list(word), list(lemma), tags.split(";")


class TagLemmatization(Lemmatization, TagSIGMORPHON2017Task1):
    pass


class Histnorm(Seq2SeqDataLoader):
    def read_file(self, file):
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                historical, modern = line.strip().split("\t")
                yield list(historical), list(modern)


class StandardG2P(Seq2SeqDataLoader):
    def read_file(self, file):
        try:
            with open(file, "r", encoding="utf-8") as fp:
                for line in fp.readlines():
                    grapheme, phoneme = line.strip().split("\t")
                    yield grapheme.split(" "), phoneme.split(" ")
        except UnicodeDecodeError:
            with open(file, "r", encoding="utf-8", errors="replace") as fp:
                for line in fp.readlines():
                    grapheme, phoneme = line.strip().split("\t")
                    yield grapheme.split(" "), phoneme.split(" ")


class StandardP2G(StandardG2P):
    def read_file(self, file):
        for grapheme, phoneme in super().read_file(file):
            yield phoneme, grapheme


class AlignStandardG2P(AlignSeq2SeqDataLoader, StandardG2P):
    pass


class Transliteration(Seq2SeqDataLoader):
    def read_file(self, file):
        root = xml.etree.ElementTree.parse(file).getroot()
        for names in root.findall("Name"):
            names = [n.text for n in names]
            src, trgs = names[0], names[1:]
            for trg in trgs:
                yield list(src), list(trg)


class AlignTransliteration(AlignSeq2SeqDataLoader, Transliteration):
    pass
