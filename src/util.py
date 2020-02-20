import logging
import math
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

tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')


class NamedEnum(Enum):
    def __str__(self):
        return self.value


def log_grad_norm(self, grad_input, grad_output, logger=None):
    try:
        logger.debug('')
        logger.debug('Inside %r backward', self.__class__.__name__)
        logger.debug('grad_input size: %r', grad_input[0].size())
        logger.debug('grad_output size: %r', grad_output[0].size())
        logger.debug('grad_input norm: %r', grad_input[0].detach().norm())
    except:
        pass


def grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.detach().norm(norm_type)
            total_norm += param_norm**norm_type
        total_norm = total_norm**(1. / norm_type)
    return total_norm


class WarmupInverseSquareRootSchedule(LambdaLR):
    """ Linear warmup and then inverse square root decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_factor = warmup_steps**0.5
        super(WarmupInverseSquareRootSchedule,
              self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.decay_factor * step**-0.5


def maybe_mkdir(filename):
    '''
    maybe mkdir
    '''
    path = os.path.dirname(filename)
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (record.levelname, time.strftime('%x %X'),
                                   timedelta(seconds=elapsed_seconds))
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def get_logger(log_file, log_level='info'):
    '''
    create logger and output to file and stdout
    '''
    assert log_level in ['info', 'debug']
    log_formatter = LogFormatter()
    logger = logging.getLogger()
    log_level = {'info': logging.INFO, 'debug': logging.DEBUG}[log_level]
    logger.setLevel(log_level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(log_formatter)
    logger.addHandler(stream)
    filep = logging.FileHandler(log_file, mode='a')
    filep.setFormatter(log_formatter)
    logger.addHandler(filep)
    return logger


def get_temp_log_filename(prefix='exp', dir='scratch/explog'):
    id = id_generator()
    fp = f'{dir}/{prefix}-{id}'
    maybe_mkdir(fp)
    return fp


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


@dataclass
class Eval:
    desc: str
    long_desc: str
    res: float


class Evaluator(object):
    def __init__(self):
        pass

    def evaluate_all(self, data_iter, nb_data, model, decode_fn) -> List[Eval]:
        raise NotImplementedError


class BasicEvaluator(Evaluator):
    '''docstring for BasicEvaluator'''
    def evaluate(self, predict, ground_truth):
        '''
        evaluate single instance
        '''
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

    def evaluate_all(self, data_iter, nb_data, model, decode_fn):
        '''
        evaluate all instances
        '''
        correct, distance, nb_sample = 0, 0, 0
        for src, trg in tqdm(data_iter(), total=nb_data):
            pred, _ = decode_fn(model, src)
            nb_sample += 1
            trg = trg.view(-1).tolist()
            trg = [x for x in trg if x != BOS_IDX and x != EOS_IDX]
            corr, dist = self.evaluate(pred, trg)
            correct += corr
            distance += dist
        acc = round(correct / nb_sample * 100, 4)
        distance = round(distance / nb_sample, 4)
        return [
            Eval('acc', 'accuracy', acc),
            Eval('dist', 'average edit distance', distance)
        ]


class HistnormEvaluator(BasicEvaluator):
    def evaluate(self, predict, ground_truth):
        correct, dist = super().evaluate(predict, ground_truth)
        return correct, dist / len(ground_truth)


class G2PEvaluator(BasicEvaluator):
    def evaluate(self, predict, ground_truth):
        correct, dist = super().evaluate(predict, ground_truth)
        return correct, dist / len(ground_truth)

    def evaluate_all(self, data_iter, nb_data, model, decode_fn):
        src_dict = defaultdict(list)
        for src, trg in tqdm(data_iter(), total=nb_data):
            pred, _ = decode_fn(model, src)
            trg = trg.view(-1).tolist()
            trg = [x for x in trg if x != BOS_IDX and x != EOS_IDX]
            corr, dist = self.evaluate(pred, trg)
            src_dict[str(src.cpu().view(-1).tolist())].append((corr, dist))
        correct, distance, nb_sample = 0, 0, 0
        for evals in src_dict.values():
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
        return [
            Eval('acc', 'accuracy', acc),
            Eval('per', 'phenome error rate', distance)
        ]


class P2GEvaluator(G2PEvaluator):
    def evaluate_all(self, data_iter, nb_data, model, decode_fn):
        results = super().evaluate_all(data_iter, nb_data, model, decode_fn)
        return [results[0], Eval('ger', 'grapheme error rate', results[1].res)]


class PairBasicEvaluator(BasicEvaluator):
    '''docstring for PairBasicEvaluator'''
    def evaluate(self, predict, ground_truth):
        '''
        evaluate single instance
        '''
        predict = [x for x in predict if x != STEP_IDX]
        ground_truth = [x for x in ground_truth if x != STEP_IDX]
        return super().evaluate(predict, ground_truth)


class PairG2PEvaluator(PairBasicEvaluator, G2PEvaluator):
    pass


class TranslitEvaluator(BasicEvaluator):
    '''docstring for TranslitEvaluator'''
    def evaluate_all(self, data_iter, nb_data, model, decode_fn):
        '''
        evaluate all instances
        '''
        def helper(src, trgs):
            pred, _ = decode_fn(model, src)
            best_corr, best_dist, closest_ref = 0, float('inf'), None
            for trg in trgs:
                trg = trg[1:-1].view(-1)
                corr, dist = self.evaluate(pred, trg)
                best_corr = max(best_corr, corr)
                if dist < best_dist:
                    best_dist = dist
                    closest_ref = trg
            lcs = (len(closest_ref) + len(pred) - best_dist) / 2
            r = lcs / len(closest_ref)
            try:
                p = lcs / len(pred)
            except:
                p = 0
            f = 2 * r * p / (r + p)
            return best_corr, f

        correct, fscore, nb_sample = 0, 0, 0
        prev_src, trgs = None, []

        for src, trg in tqdm(data_iter(), total=nb_data):
            if prev_src is not None and self.evaluate(
                    src.view(-1), prev_src.view(-1))[0] == 0:
                corr, f = helper(prev_src, trgs)
                correct += corr
                fscore += f
                nb_sample += 1
                prev_src = src
                trgs = [trg]
            else:
                prev_src = src
                trgs.append(trg)
        corr, f = helper(prev_src, trgs)
        correct += corr
        fscore += f
        nb_sample += 1

        acc = round(correct / nb_sample * 100, 4)
        mean_fscore = round(fscore / nb_sample, 4)
        return [
            Eval('acc', 'accuracy', acc),
            Eval('meanfs', 'mean F-score', mean_fscore)
        ]


class PairTranslitEvaluator(PairBasicEvaluator, TranslitEvaluator):
    pass


def edit_distance(str1, str2):
    '''Simple Levenshtein implementation for evalm.'''
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
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + dg)
    return int(table[len(str2)][len(str1)])
