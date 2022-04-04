import logging
import os
import random
import string
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial
from typing import List

import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from dataloader import BOS_IDX, EOS_IDX, STEP_IDX

tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")


class NamedEnum(Enum):
    def __str__(self):
        return self.value


def grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.detach().norm(norm_type)
            total_norm += param_norm**norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


class WarmupInverseSquareRootSchedule(LambdaLR):
    """Linear warmup and then inverse square root decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_factor = warmup_steps**0.5
        super(WarmupInverseSquareRootSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.decay_factor * step**-0.5


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


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def get_logger(log_file, log_level="info"):
    """
    create logger and output to file and stdout
    """
    assert log_level in ["info", "debug"]
    log_formatter = LogFormatter()
    logger = logging.getLogger()
    log_level = {"info": logging.INFO, "debug": logging.DEBUG}[log_level]
    logger.setLevel(log_level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(log_formatter)
    logger.addHandler(stream)
    filep = logging.FileHandler(log_file, mode="a")
    filep.setFormatter(log_formatter)
    logger.addHandler(filep)
    return logger


def get_temp_log_filename(prefix="exp", dir="scratch/explog"):
    id = id_generator()
    fp = f"{dir}/{prefix}-{id}"
    maybe_mkdir(fp)
    return fp


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def unpack_batch(batch):
    if isinstance(batch, list) and isinstance(batch[0], list):
        return [
            [char for char in seq if char != BOS_IDX and char != EOS_IDX]
            for seq in batch
        ]
    batch = batch.transpose(0, 1).cpu().numpy()
    bs, seq_len = batch.shape
    output = []
    for i in range(bs):
        seq = []
        for j in range(seq_len):
            elem = batch[i, j]
            if elem == BOS_IDX:
                continue
            if elem == EOS_IDX:
                break
            seq.append(elem)
        output.append(seq)
    return output


@dataclass
class Eval:
    desc: str
    long_desc: str
    res: float


class Evaluator(object):
    def __init__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def evaluate(self, predict, ground_truth):
        raise NotImplementedError

    def add(self, source, predict, target):
        raise NotImplementedError

    def compute(self, reset=True) -> List[Eval]:
        raise NotImplementedError

    def evaluate_all(
        self, data_iter, batch_size, nb_data, model, decode_fn
    ) -> List[Eval]:
        for src, src_mask, trg, trg_mask in tqdm(data_iter(batch_size), total=nb_data):
            pred, _ = decode_fn(model, src, src_mask)
            self.add(src, pred, trg)
        return self.compute(reset=True)


class BasicEvaluator(Evaluator):
    """docstring for BasicEvaluator"""

    def __init__(self):
        self.correct = 0
        self.distance = 0
        self.nb_sample = 0

    def reset(self):
        self.correct = 0
        self.distance = 0
        self.nb_sample = 0

    def evaluate(self, predict, ground_truth):
        """
        evaluate single instance
        """
        correct = 1
        if len(predict) == len(ground_truth):
            for elem1, elem2 in zip(predict, ground_truth):
                if elem1 != elem2:
                    correct = 0
                    break
        else:
            correct = 0
        dist = edit_distance(predict, ground_truth)
        return correct, dist

    def add(self, source, predict, target):
        predict = unpack_batch(predict)
        target = unpack_batch(target)
        for p, t in zip(predict, target):
            correct, distance = self.evaluate(p, t)
            self.correct += correct
            self.distance += distance
            self.nb_sample += 1

    def compute(self, reset=True):
        accuracy = round(self.correct / self.nb_sample * 100, 4)
        distance = round(self.distance / self.nb_sample, 4)
        if reset:
            self.reset()
        return [
            Eval("acc", "accuracy", accuracy),
            Eval("dist", "average edit distance", distance),
        ]


class HistnormEvaluator(BasicEvaluator):
    def evaluate(self, predict, ground_truth):
        correct, dist = super().evaluate(predict, ground_truth)
        return correct, dist / len(ground_truth)


class G2PEvaluator(BasicEvaluator):
    def __init__(self):
        self.src_dict = defaultdict(list)

    def reset(self):
        self.src_dict = defaultdict(list)

    def evaluate(self, predict, ground_truth):
        correct, dist = super().evaluate(predict, ground_truth)
        return correct, dist / len(ground_truth)

    def add(self, source, predict, target):
        source = unpack_batch(source)
        predict = unpack_batch(predict)
        target = unpack_batch(target)
        for s, p, t in zip(source, predict, target):
            correct, distance = self.evaluate(p, t)
            self.src_dict[str(s)].append((correct, distance))

    def compute(self, reset=True):
        correct, distance, nb_sample = 0, 0, 0
        for evals in self.src_dict.values():
            corr, dist = evals[0]
            for c, d in evals:
                if c > corr:
                    corr = c
                if d < dist:
                    dist = d
            correct += corr
            distance += dist
            nb_sample += 1
        acc = round(correct / nb_sample * 100, 4)
        distance = round(distance / nb_sample, 4)
        if reset:
            self.reset()
        return [
            Eval("acc", "accuracy", acc),
            Eval("per", "phenome error rate", distance),
        ]


class P2GEvaluator(G2PEvaluator):
    def compute(self, reset=True):
        results = super().compute(reset=reset)
        return [results[0], Eval("ger", "grapheme error rate", results[1].res)]


class PairBasicEvaluator(BasicEvaluator):
    """docstring for PairBasicEvaluator"""

    def evaluate(self, predict, ground_truth):
        """
        evaluate single instance
        """
        predict = [x for x in predict if x != STEP_IDX]
        ground_truth = [x for x in ground_truth if x != STEP_IDX]
        return super().evaluate(predict, ground_truth)


class PairG2PEvaluator(PairBasicEvaluator, G2PEvaluator):
    pass


class TranslitEvaluator(BasicEvaluator):
    """docstring for TranslitEvaluator"""

    def __init__(self):
        self.src_dict = defaultdict(list)

    def reset(self):
        self.src_dict = defaultdict(list)

    def add(self, source, predict, target):
        source = unpack_batch(source)
        predict = unpack_batch(predict)
        target = unpack_batch(target)
        for s, p, t in zip(source, predict, target):
            correct, distance = self.evaluate(p, t)
            self.src_dict[str(s)].append((correct, distance, len(p), len(t)))

    def compute(self, reset=True):
        correct, fscore, nb_sample = 0, 0, 0
        for evals in self.src_dict.values():
            corr, dist, pred_len, trg_len = evals[0]
            for c, d, pl, tl in evals:
                if c > corr:
                    corr = c
                if d < dist:
                    dist = d
                    pred_len = pl
                    trg_len = tl
            lcs = (trg_len + pred_len - dist) / 2
            r = lcs / trg_len
            try:
                p = lcs / pred_len
            except ZeroDivisionError:
                p = 0
            f = 2 * r * p / (r + p)

            correct += corr
            fscore += f
            nb_sample += 1

        acc = round(correct / nb_sample * 100, 4)
        mean_fscore = round(fscore / nb_sample, 4)
        if reset:
            self.reset()
        return [
            Eval("acc", "accuracy", acc),
            Eval("meanfs", "mean F-score", mean_fscore),
        ]


class PairTranslitEvaluator(PairBasicEvaluator, TranslitEvaluator):
    pass


def edit_distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(
                table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + dg
            )
    return int(table[len(str2)][len(str1)])
