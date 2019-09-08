import glob
import os
import random
import re
from functools import partial
from math import ceil
from typing import List

import numpy as np
import torch
import util
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaseTrainer(object):
    '''docstring for Trainer.'''

    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.data = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.min_lr = 0
        self.scheduler = None
        self.evaluator = None
        self.last_devloss = float('inf')
        self.models = list()

    def checklist_before_run(self):
        assert self.data is not None, 'call load_data before run'
        assert self.model is not None, 'call build_model before run'
        assert self.optimizer is not None, 'call setup_training before run'
        assert self.scheduler is not None, 'call setup_scheduler before run'
        assert self.evaluator is not None, 'call setup_evalutator before run'

    def load_data(self, dataset, train, dev, test, opt):
        raise NotImplementedError

    def build_model(self, opt):
        raise NotImplementedError

    def load_model(self, model):
        assert self.model is None
        self.logger.info('load model in %s', model)
        self.model = torch.load(model, map_location=self.device)
        self.model = self.model.to(self.device)
        epoch = int(model.split('_')[-1])
        return epoch

    def smart_load_model(self, model_prefix):
        assert self.model is None
        models = []
        for model in glob.glob(f'{model_prefix}.nll*'):
            res = re.findall(r'\w*_\d+\.?\d*', model)
            loss_ = res[0].split('_')
            evals_ = res[1:-1]
            epoch_ = res[-1].split('_')
            assert loss_[0] == 'nll' and epoch_[0] == 'epoch'
            loss, epoch = float(loss_[1]), int(epoch_[1])
            evals = []
            for ev in evals_:
                ev = ev.split('_')
                evals.append(util.Eval(ev[0], ev[0], float(ev[1])))
            models.append((epoch, (model, loss, evals)))
        self.models = [x[1] for x in sorted(models)]
        return self.load_model(self.models[-1][0])

    def setup_training(self, optimizer, lr, momentum):
        assert self.model is not None
        optimizer = optimizer.lower()
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr, momentum=momentum)
        elif optimizer == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        elif optimizer == 'amsgrad':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr, amsgrad=True)
        else:
            raise ValueError

    def setup_scheduler(self, min_lr, patience, cooldown, discount_factor):
        self.min_lr = min_lr
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            'min',
            patience=patience,
            cooldown=cooldown,
            factor=discount_factor,
            min_lr=min_lr)

    def save_training(self, model_fp):
        save_objs = (self.optimizer.state_dict(), self.scheduler.state_dict())
        torch.save(save_objs, f'{model_fp}.progress')

    def load_training(self, model_fp):
        assert self.model is not None
        optimizer_state, scheduler_state = torch.load(f'{model_fp}.progress')
        self.optimizer.load_state_dict(optimizer_state)
        self.scheduler.load_state_dict(scheduler_state)

    def setup_evalutator(self):
        raise NotImplementedError

    def get_lr(self):
        try:
            return self.scheduler.get_lr()[0]
        except:
            assert isinstance(self.scheduler, ReduceLROnPlateau)
            return self.optimizer.param_groups[0]['lr']

    def train(self, epoch_idx, batch_size, max_norm):
        logger, model, data = self.logger, self.model, self.data
        logger.info('At %d-th epoch with lr %f.', epoch_idx, self.get_lr())
        model.train()
        sampler, nb_batch = self.iterate_batch(TRAIN, batch_size)
        losses, cnt = 0, 0
        for batch in tqdm(sampler(batch_size), total=nb_batch):
            loss = model.get_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            logger.debug('loss %f with total grad norm %f', loss,
                         util.grad_norm(model.parameters()))
            self.optimizer.step()
            losses += loss.item()
            cnt += 1
        loss = losses / cnt
        self.logger.info(
            f'Running average train loss is {loss} at epoch {epoch_idx}')
        return loss

    def iterate_batch(self, mode, batch_size):
        if mode == TRAIN:
            return self.data.train_batch_sample, ceil(
                self.data.nb_train / batch_size)
        elif mode == DEV:
            return self.data.dev_batch_sample, ceil(
                self.data.nb_dev / batch_size)
        elif mode == TEST:
            return self.data.test_batch_sample, ceil(
                self.data.nb_test / batch_size)
        else:
            raise ValueError(f'wrong mode: {mode}')

    def calc_loss(self, mode, batch_size, epoch_idx) -> float:
        self.model.eval()
        sampler, nb_batch = self.iterate_batch(mode, batch_size)
        loss, cnt = 0., 0
        for batch in tqdm(sampler(batch_size), total=nb_batch):
            loss += self.model.get_loss(batch).item()
            cnt += 1
        loss = loss / cnt
        self.logger.info(f'Average {mode} loss is {loss} at epoch {epoch_idx}')
        return loss

    def iterate_instance(self, mode):
        if mode == TRAIN:
            return self.data.train_sample, self.data.nb_train
        elif mode == DEV:
            return self.data.dev_sample, self.data.nb_dev
        elif mode == TEST:
            return self.data.test_sample, self.data.nb_test
        else:
            raise ValueError(f'wrong mode: {mode}')

    def evaluate(self, mode, epoch_idx, decode_fn) -> List[util.Eval]:
        raise NotImplementedError

    def decode(self, mode, write_fp, decode_fn):
        raise NotImplementedError

    def update_lr_and_stop_early(self, epoch_idx, devloss, estop):
        prev_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(devloss)
        curr_lr = self.optimizer.param_groups[0]['lr']

        stop_early = True
        if (self.last_devloss - devloss) < estop and \
            prev_lr == curr_lr == self.min_lr:
            self.logger.info(
                'Early stopping triggered with epoch %d (previous dev loss: %f, current: %f)',
                epoch_idx, self.last_devloss, devloss)
            stop_status = stop_early
        else:
            stop_status = not stop_early
        self.last_devloss = devloss
        return stop_status

    def save_model(self, epoch_idx, devloss, eval_res, model_fp):
        eval_tag = ''.join(['{}_{}.'.format(e.desc, e.res) for e in eval_res])
        fp = f'{model_fp}.nll_{devloss:.4f}.{eval_tag}epoch_{epoch_idx}'
        torch.save(self.model, fp)
        self.models.append((fp, devloss, eval_res))

    def select_model(self, opt):
        raise NotImplementedError

    def reload_and_test(self, model_fp, best_fp, bs, decode_fn):
        self.model = None
        self.logger.info(f'loading {best_fp} for testing')
        self.load_model(best_fp)
        self.logger.info('decoding dev set')
        self.decode(DEV, f'{model_fp}.decode', decode_fn)
        if self.data.test_file is not None:
            self.calc_loss(TEST, bs, -1)
            self.logger.info('decoding test set')
            self.decode(TEST, f'{model_fp}.decode', decode_fn)
            results = self.evaluate(TEST, -1, decode_fn)
            if results:
                results = ' '.join([f'{r.desc} {r.res}' for r in results])
                self.logger.info(f'TEST {model_fp.split("/")[-1]} {results}')

    def cleanup(self, saveall, save_fps, model_fp):
        if not saveall:
            for model in self.models:
                fp = model[0]
                if fp in save_fps:
                    continue
                os.remove(fp)
        os.remove(f'{model_fp}.progress')

    def run(self, opt, start_epoch, decode_fn=None):
        '''
        helper for training
        '''
        self.checklist_before_run()
        finish = False
        for epoch_idx in range(start_epoch, start_epoch + opt.epochs):
            self.train(epoch_idx, opt.bs, opt.max_norm)
            with torch.no_grad():
                devloss = self.calc_loss(DEV, opt.bs, epoch_idx)
                eval_res = self.evaluate(DEV, epoch_idx, decode_fn)
            if self.update_lr_and_stop_early(epoch_idx, devloss, opt.estop):
                finish = True
                break
            self.save_model(epoch_idx, devloss, eval_res, opt.model)
            self.save_training(opt.model)
        if finish or opt.cleanup_anyway:
            best_fp, save_fps = self.select_model(opt)
            with torch.no_grad():
                self.reload_and_test(opt.model, best_fp, opt.bs, decode_fn)
            self.cleanup(opt.saveall, save_fps, opt.model)