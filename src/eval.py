"""
train
"""
import os
from functools import partial

import torch
from tqdm import tqdm

import dataloader
import model
# import transformer
import util
from decoding import get_decode_fn
from model import dummy_mask
from trainer import DEV, TEST
from train import Trainer

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")


class Evaluator(Trainer):

    def evaluate(self, mode, epoch_idx, decode_fn):
        self.model.eval()
        sampler, nb_instance = self.iterate_instance(mode)
        decode_fn.reset()
        results = self.evaluator.evaluate_all(
            sampler, nb_instance, self.model, decode_fn
        )
        decode_fn.reset()
        for result in results:
            self.logger.info(
                f"{mode} {result.long_desc} is {result.res} at epoch {epoch_idx}"
            )
        return results

    def decode(self, mode, write_fp, decode_fn):
        self.model.eval()
        cnt = 0
        sampler, nb_instance = self.iterate_instance(mode)
        decode_fn.reset()
        with open(f"{write_fp}.{mode}.tsv", "w") as fp:
            fp.write("prediction\ttarget\tloss\tdist\n")
            for src, trg in tqdm(sampler(), total=nb_instance):
                pred, _ = decode_fn(self.model, src)
                dist = util.edit_distance(pred, trg.view(-1).tolist()[1:-1])

                src_mask = dummy_mask(src)
                trg_mask = dummy_mask(trg)
                data = (src, src_mask, trg, trg_mask)
                loss = self.model.get_loss(data).item()

                trg = self.data.decode_target(trg)[1:-1]
                pred = self.data.decode_target(pred)
                fp.write(f'{" ".join(pred)}\t{" ".join(trg)}\t{loss}\t{dist}\n')
                cnt += 1
        decode_fn.reset()
        self.logger.info(f"finished decoding {cnt} {mode} instance")

    def checklist_before_run(self):
        assert self.data is not None, "call load_data before run"
        assert self.model is not None, "call build_model before run"

    def test(self, bs, decode_fn, model_fp):
        self.calc_loss(DEV, bs, -1)
        self.logger.info("decoding dev set")
        self.decode(DEV, f"{model_fp}.decode", decode_fn)
        results = self.evaluate(DEV, -1, decode_fn)
        if results:
            results = " ".join([f"{r.desc} {r.res}" for r in results])
            self.logger.info(f'DEV {results}')

        self.calc_loss(TEST, bs, -1)
        self.logger.info("decoding test set")
        self.decode(TEST, f"{model_fp}.decode", decode_fn)
        results = self.evaluate(TEST, -1, decode_fn)
        if results:
            results = " ".join([f"{r.desc} {r.res}" for r in results])
            self.logger.info(f'TEST {results}')

    def run(self, start_epoch, decode_fn=None):
        self.checklist_before_run()
        params = self.params

        with torch.no_grad():
            self.test(params.bs, decode_fn, params.model)


def main():
    trainer = Evaluator()
    params = trainer.params
    assert params.load and params.load != "0"

    decode_fn = get_decode_fn(
        params.decode, params.max_decode_len, params.decode_beam_size
    )
    trainer.load_data(params.dataset, params.train, params.dev, params.test)
    trainer.setup_evalutator()

    if params.load == "smart":
        start_epoch = trainer.smart_load_model(params.model) + 1
    else:
        start_epoch = trainer.load_model(params.load) + 1

    trainer.run(start_epoch, decode_fn=decode_fn)


if __name__ == "__main__":
    main()
