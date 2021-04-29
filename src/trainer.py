import argparse
import glob
import os
import random
import re
from dataclasses import dataclass
from functools import partial
from math import ceil
from typing import List, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import util

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")

TRAIN = "train"
DEV = "dev"
TEST = "test"


class Optimizer(util.NamedEnum):
    sgd = "sgd"
    adadelta = "adadelta"
    adam = "adam"
    amsgrad = "amsgrad"


class Scheduler(util.NamedEnum):
    reducewhenstuck = "reducewhenstuck"
    warmupinvsqr = "warmupinvsqr"


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Evaluation:
    filepath: str
    devloss: float
    evaluation_result: Optional[List[util.Eval]]


class BaseTrainer(object):
    """docstring for Trainer."""

    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser()
        self.set_args()
        self.params = self.get_params()

        util.maybe_mkdir(self.params.model)
        self.logger = util.get_logger(
            self.params.model + ".log", log_level=self.params.loglevel
        )
        for key, value in vars(self.params).items():
            self.logger.info("command line argument: %s - %r", key, value)
        setup_seed(self.params.seed)

        self.data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.min_lr = 0
        self.scheduler = None
        self.evaluator = None
        self.global_steps = 0
        self.last_devloss = float("inf")
        self.models: List[Evaluation] = list()

    def set_args(self):
        """
        get_args
        """
        # fmt: off
        parser = self.parser
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--train', required=True, type=str, nargs='+')
        parser.add_argument('--dev', required=True, type=str, nargs='+')
        parser.add_argument('--test', default=None, type=str, nargs='+')
        parser.add_argument('--model', required=True, help='dump model filename')
        parser.add_argument('--load', default='', help='load model and continue training; with `smart`, recover training automatically')
        parser.add_argument('--bs', default=20, type=int, help='training batch size')
        parser.add_argument('--epochs', default=20, type=int, help='maximum training epochs')
        parser.add_argument('--max_steps', default=0, type=int, help='maximum training steps')
        parser.add_argument('--warmup_steps', default=4000, type=int, help='number of warm up steps')
        parser.add_argument('--total_eval', default=-1, type=int, help='total number of evaluation')
        parser.add_argument('--optimizer', default=Optimizer.adam, type=Optimizer, choices=list(Optimizer))
        parser.add_argument('--scheduler', default=Scheduler.reducewhenstuck, type=Scheduler, choices=list(Scheduler))
        parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
        parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD')
        parser.add_argument('--beta1', default=0.9, type=float, help='beta1 of Adam')
        parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of Adam')
        parser.add_argument('--estop', default=1e-8, type=float, help='early stopping criterion')
        parser.add_argument('--cooldown', default=0, type=int, help='cooldown of `ReduceLROnPlateau`')
        parser.add_argument('--patience', default=0, type=int, help='patience of `ReduceLROnPlateau`')
        parser.add_argument('--discount_factor', default=0.5, type=float, help='discount factor of `ReduceLROnPlateau`')
        parser.add_argument('--max_norm', default=0, type=float, help='gradient clipping max norm')
        parser.add_argument('--gpuid', default=[], nargs='+', type=int, help='choose which GPU to use')
        parser.add_argument('--loglevel', default='info', choices=['info', 'debug'])
        parser.add_argument('--saveall', default=False, action='store_true', help='keep all models')
        parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle the data')
        parser.add_argument('--cleanup_anyway', default=False, action='store_true', help='cleanup anyway')
        # fmt: on

    def get_params(self):
        return self.parser.parse_args()

    def checklist_before_run(self):
        assert self.data is not None, "call load_data before run"
        assert self.model is not None, "call build_model before run"
        assert self.optimizer is not None, "call setup_training before run"
        assert self.scheduler is not None, "call setup_scheduler before run"
        assert self.evaluator is not None, "call setup_evalutator before run"

    def load_data(self, dataset, train, dev, test):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def load_model(self, model):
        assert self.model is None
        self.logger.info("load model in %s", model)
        self.model = torch.load(model, map_location=self.device)
        self.model = self.model.to(self.device)
        epoch = int(model.split("_")[-1])
        return epoch

    def smart_load_model(self, model_prefix):
        assert self.model is None
        models = []
        for model in glob.glob(f"{model_prefix}.nll*"):
            res = re.findall(r"\w*_\d+\.?\d*", model[len(model_prefix) :])
            loss_ = res[0].split("_")
            evals_ = res[1:-1]
            epoch_ = res[-1].split("_")
            assert loss_[0] == "nll" and epoch_[0] == "epoch"
            loss, epoch = float(loss_[1]), int(epoch_[1])
            evals = []
            for ev in evals_:
                ev = ev.split("_")
                evals.append(util.Eval(ev[0], ev[0], float(ev[1])))
            models.append((epoch, Evaluation(model, loss, evals)))
        self.models = [x[1] for x in sorted(models)]
        return self.load_model(self.models[-1].filepath)

    def setup_training(self):
        assert self.model is not None
        params = self.params
        if params.optimizer == Optimizer.sgd:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), params.lr, momentum=params.momentum
            )
        elif params.optimizer == Optimizer.adadelta:
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), params.lr)
        elif params.optimizer == Optimizer.adam:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), params.lr, betas=(params.beta1, params.beta2)
            )
        elif params.optimizer == Optimizer.amsgrad:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                params.lr,
                betas=(params.beta1, params.beta2),
                amsgrad=True,
            )
        else:
            raise ValueError

        self.min_lr = params.min_lr
        if params.scheduler == Scheduler.reducewhenstuck:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                "min",
                patience=params.patience,
                cooldown=params.cooldown,
                factor=params.discount_factor,
                min_lr=params.min_lr,
            )
        elif params.scheduler == Scheduler.warmupinvsqr:
            self.scheduler = util.WarmupInverseSquareRootSchedule(
                self.optimizer, params.warmup_steps
            )
        else:
            raise ValueError

    def save_training(self, model_fp):
        save_objs = (self.optimizer.state_dict(), self.scheduler.state_dict())
        torch.save(save_objs, f"{model_fp}.progress")

    def load_training(self, model_fp):
        assert self.model is not None
        if os.path.isfile(f"{model_fp}.progress"):
            optimizer_state, scheduler_state = torch.load(f"{model_fp}.progress")
            self.optimizer.load_state_dict(optimizer_state)
            self.scheduler.load_state_dict(scheduler_state)
        else:
            self.logger.warning("cannot find optimizer & scheduler file")

    def setup_evalutator(self):
        raise NotImplementedError

    def get_lr(self):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            return self.optimizer.param_groups[0]["lr"]
        try:
            return self.scheduler.get_last_lr()[0]
        except AttributeError:
            return self.scheduler.get_lr()[0]

    def train(self, epoch_idx, batch_size, max_norm):
        logger, model = self.logger, self.model
        logger.info("At %d-th epoch with lr %f.", epoch_idx, self.get_lr())
        model.train()
        sampler, nb_batch = self.iterate_batch(TRAIN, batch_size)
        losses, cnt = 0, 0
        for batch in tqdm(sampler(batch_size), total=nb_batch):
            loss = model.get_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            logger.debug(
                "loss %f with total grad norm %f",
                loss,
                util.grad_norm(model.parameters()),
            )
            self.optimizer.step()
            if not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()
            self.global_steps += 1
            losses += loss.item()
            cnt += 1
        loss = losses / cnt
        self.logger.info(f"Running average train loss is {loss} at epoch {epoch_idx}")
        return loss

    def iterate_batch(self, mode, batch_size):
        if mode == TRAIN:
            return (self.data.train_batch_sample, ceil(self.data.nb_train / batch_size))
        elif mode == DEV:
            return (self.data.dev_batch_sample, ceil(self.data.nb_dev / batch_size))
        elif mode == TEST:
            return (self.data.test_batch_sample, ceil(self.data.nb_test / batch_size))
        else:
            raise ValueError(f"wrong mode: {mode}")

    def calc_loss(self, mode, batch_size, epoch_idx) -> float:
        self.model.eval()
        sampler, nb_batch = self.iterate_batch(mode, batch_size)
        loss, cnt = 0.0, 0
        for batch in tqdm(sampler(batch_size), total=nb_batch):
            loss += self.model.get_loss(batch).item()
            cnt += 1
        loss = loss / cnt
        self.logger.info(f"Average {mode} loss is {loss} at epoch {epoch_idx}")
        return loss

    def iterate_instance(self, mode):
        if mode == TRAIN:
            return self.data.train_sample, self.data.nb_train
        elif mode == DEV:
            return self.data.dev_sample, self.data.nb_dev
        elif mode == TEST:
            return self.data.test_sample, self.data.nb_test
        else:
            raise ValueError(f"wrong mode: {mode}")

    def evaluate(self, mode, batch_size, epoch_idx, decode_fn) -> List[util.Eval]:
        raise NotImplementedError

    def decode(self, mode, batch_size, write_fp, decode_fn) -> List[util.Eval]:
        raise NotImplementedError

    def update_lr_and_stop_early(self, epoch_idx, devloss, estop):
        stop_early = True

        if isinstance(self.scheduler, ReduceLROnPlateau):
            prev_lr = self.get_lr()
            self.scheduler.step(devloss)
            curr_lr = self.get_lr()

            if (
                self.last_devloss - devloss
            ) < estop and prev_lr == curr_lr == self.min_lr:
                self.logger.info(
                    "Early stopping triggered with epoch %d (previous dev loss: %f, current: %f)",
                    epoch_idx,
                    self.last_devloss,
                    devloss,
                )
                stop_status = stop_early
            else:
                stop_status = not stop_early
            self.last_devloss = devloss
        else:
            stop_status = not stop_early
        return stop_status

    def save_model(
        self, epoch_idx, devloss: float, eval_res: List[util.Eval], model_fp
    ):
        eval_tag = "".join(["{}_{}.".format(e.desc, e.res) for e in eval_res])
        fp = f"{model_fp}.nll_{devloss:.4f}.{eval_tag}epoch_{epoch_idx}"
        torch.save(self.model, fp)
        self.models.append(Evaluation(fp, devloss, eval_res))

    def select_model(self):
        raise NotImplementedError

    def reload_and_test(self, model_fp, best_fp, batch_size, decode_fn):
        self.model = None
        self.logger.info(f"loading {best_fp} for testing")
        self.load_model(best_fp)
        self.calc_loss(DEV, batch_size, -1)
        self.logger.info("decoding dev set")
        results = self.decode(DEV, batch_size, f"{model_fp}.decode", decode_fn)
        if results:
            for result in results:
                self.logger.info(f"DEV {result.long_desc} is {result.res} at epoch -1")
            results = " ".join([f"{r.desc} {r.res}" for r in results])
            self.logger.info(f'DEV {model_fp.split("/")[-1]} {results}')

        if self.data.test_file is not None:
            self.calc_loss(TEST, batch_size, -1)
            self.logger.info("decoding test set")
            results = self.decode(TEST, batch_size, f"{model_fp}.decode", decode_fn)
            if results:
                for result in results:
                    self.logger.info(
                        f"TEST {result.long_desc} is {result.res} at epoch -1"
                    )
                results = " ".join([f"{r.desc} {r.res}" for r in results])
                self.logger.info(f'TEST {model_fp.split("/")[-1]} {results}')

    def cleanup(self, saveall, save_fps, model_fp):
        if not saveall:
            for model in self.models:
                if model.filepath in save_fps:
                    continue
                os.remove(model.filepath)
        os.remove(f"{model_fp}.progress")

    def run(self, start_epoch, decode_fn=None):
        """
        helper for training
        """
        self.checklist_before_run()
        finish = False
        params = self.params
        steps_per_epoch = ceil(self.data.nb_train / params.bs)
        if params.max_steps > 0:
            max_epochs = ceil(params.max_steps / steps_per_epoch)
        else:
            max_epochs = params.epochs
        params.max_steps = max_epochs * steps_per_epoch
        self.logger.info(
            f"maximum training {params.max_steps} steps ({max_epochs} epochs)"
        )
        if params.total_eval > 0:
            eval_every = max(max_epochs // params.total_eval, 1)
        else:
            eval_every = 1
        self.logger.info(f"evaluate every {eval_every} epochs")
        for epoch_idx in range(start_epoch, max_epochs):
            self.train(epoch_idx, params.bs, params.max_norm)
            if not (
                epoch_idx
                and (epoch_idx % eval_every == 0 or epoch_idx + 1 == max_epochs)
            ):
                continue
            with torch.no_grad():
                devloss = self.calc_loss(DEV, params.bs, epoch_idx)
                eval_res = self.evaluate(DEV, params.bs, epoch_idx, decode_fn)
            if self.update_lr_and_stop_early(epoch_idx, devloss, params.estop):
                finish = True
                break
            self.save_model(epoch_idx, devloss, eval_res, params.model)
            self.save_training(params.model)
        if finish or params.cleanup_anyway:
            best_fp, save_fps = self.select_model()
            with torch.no_grad():
                self.reload_and_test(params.model, best_fp, params.bs, decode_fn)
            self.cleanup(params.saveall, save_fps, params.model)
