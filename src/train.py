'''
train
'''
import argparse
import os
from functools import partial

import dataloader
import model
import torch
import util
from decoding import Decode, get_decode_fn
from model import dummy_mask
from tqdm import tqdm
from trainer import BaseTrainer, setup_seed

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')


class Data(util.NamedEnum):
    g2p = 'g2p'
    p2g = 'p2g'
    news15 = 'news15'
    sigmorphon16task1 = 'sigmorphon16task1'
    sigmorphon17task1 = 'sigmorphon17task1'
    sigmorphon19task1 = 'sigmorphon19task1'
    sigmorphon19task2 = 'sigmorphon19task2'
    lemma = 'lemma'
    lemmanotag = 'lemmanotag'
    lematus = 'lematus'
    unimorph = 'unimorph'


class Arch(util.NamedEnum):
    soft = 'soft'  # soft attention without input-feeding
    hard = 'hard'  # hard attention with dynamic programming without input-feeding
    approxihard = 'approxihard'  # hard attention with REINFORCE approximation without input-feeding
    softinputfeed = 'softinputfeed'  # soft attention with input-feeding
    largesoftinputfeed = 'largesoftinputfeed'  # soft attention with uncontrolled input-feeding
    approxihardinputfeed = 'approxihardinputfeed'  # hard attention with REINFORCE approximation with input-feeding
    hardmono = 'hardmono'  # hard monotonic attention
    hmm = 'hmm'  # 0th-order hard attention without input-feeding
    hmmfull = 'hmmfull'  # 1st-order hard attention without input-feeding


def get_args():
    '''
    get_args
    '''
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', required=True, type=Data, choices=list(Data))
    parser.add_argument('--train', required=True, nargs='+')
    parser.add_argument('--dev', required=True)
    parser.add_argument('--test', default=None, type=str)
    parser.add_argument('--src_lang', default=None, type=str)
    parser.add_argument('--trg_lang', default=None, type=str)
    parser.add_argument('--vocab', default=None, type=str)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--max_decode_len', default=128, type=int)
    parser.add_argument('--model', required=True, help='dump model filename')
    parser.add_argument('--load', default='', help='load model and continue training; with `smart`, recover training automatically')
    parser.add_argument('--init', default='', help='control initialization')
    parser.add_argument('--bs', default=20, type=int, help='training batch size')
    parser.add_argument('--epochs', default=20, type=int, help='maximum training epochs')
    parser.add_argument('--optimizer', default='Adam', choices=['SGD', 'Adadelta', 'Adam'])
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD')
    parser.add_argument('--estop', default=1e-8, type=float, help='early stopping criterion')
    parser.add_argument('--cooldown', default=0, type=int, help='cooldown of `ReduceLROnPlateau`')
    parser.add_argument('--patience', default=0, type=int, help='patience of `ReduceLROnPlateau`')
    parser.add_argument('--discount_factor', default=0.5, type=float, help='discount factor of `ReduceLROnPlateau`')
    parser.add_argument('--max_norm', default=0, type=float, help='gradient clipping max norm')
    parser.add_argument('--gpuid', default=[], nargs='+', type=int, help='choose which GPU to use')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout prob')
    parser.add_argument('--embed_dim', default=100, type=int, help='embedding dimension')
    parser.add_argument('--src_layer', default=1, type=int, help='source encoder number of layers')
    parser.add_argument('--trg_layer', default=1, type=int, help='target decoder number of layers')
    parser.add_argument('--src_hs', default=200, type=int, help='source encoder hidden dimension')
    parser.add_argument('--trg_hs', default=200, type=int, help='target decoder hidden dimension')
    parser.add_argument('--arch', required=True, type=Arch, choices=list(Arch))
    parser.add_argument('--nb_sample', default=2, type=int, help='number of sample in REINFORCE approximation')
    parser.add_argument('--wid_siz', default=11, type=int, help='maximum transition in 1st-order hard attention')
    parser.add_argument('--loglevel', default='info', choices=['info', 'debug'])
    parser.add_argument('--saveall', default=False, action='store_true', help='keep all models')
    parser.add_argument('--indtag', default=False, action='store_true', help='separate tag from source string')
    parser.add_argument('--decode', default=Decode.greedy, type=Decode, choices=list(Decode))
    parser.add_argument('--mono', default=False, action='store_true', help='enforce monotonicity')
    parser.add_argument('--bow', default=False, action='store_true', help='use bag-of-word encoder')
    parser.add_argument('--inputfeed', default=False, action='store_true', help='use inputfeed decoder')
    parser.add_argument('--bestacc', default=False, action='store_true', help='select model by accuracy only')
    parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle the data')
    parser.add_argument('--cleanup_anyway', default=False, action='store_true', help='cleanup anyway')
    # yapf: enable
    return parser.parse_args()


class Trainer(BaseTrainer):
    '''docstring for Trainer.'''

    def load_data(self, dataset, train, dev, test, opt):
        assert self.data is None
        logger = self.logger
        # yapf: disable
        if opt.arch == Arch.hardmono:
            if dataset == Data.sigmorphon17task1:
                self.data = dataloader.AlignSIGMORPHON2017Task1(train, dev, test, opt.shuffle)
            elif dataset == Data.g2p:
                self.data = dataloader.AlignStandardG2P(train, dev, test, opt.shuffle)
            elif dataset == Data.news15:
                self.data = dataloader.AlignTransliteration(train, dev, test, opt.shuffle)
            else:
                raise ValueError
        else:
            if dataset == Data.sigmorphon17task1:
                if opt.indtag:
                    self.data = dataloader.TagSIGMORPHON2017Task1(train, dev, test, opt.shuffle)
                else:
                    self.data = dataloader.SIGMORPHON2017Task1(train, dev, test, opt.shuffle)
            elif dataset == Data.unimorph:
                if opt.indtag:
                    self.data = dataloader.TagUnimorph(train, dev, test, opt.shuffle)
                else:
                    self.data = dataloader.Unimorph(train, dev, test, opt.shuffle)
            elif dataset == Data.sigmorphon19task1:
                assert isinstance(train, list) and len(train) == 2 and opt.indtag
                self.data = dataloader.TagSIGMORPHON2019Task1(train, dev, test, opt.shuffle)
            elif dataset == Data.sigmorphon19task2:
                assert opt.indtag
                self.data = dataloader.TagSIGMORPHON2019Task2(train, dev, test, opt.shuffle)
            elif dataset == Data.g2p:
                self.data = dataloader.StandardG2P(train, dev, test, opt.shuffle)
            elif dataset == Data.p2g:
                self.data = dataloader.StandardP2G(train, dev, test, opt.shuffle)
            elif dataset == Data.news15:
                self.data = dataloader.Transliteration(train, dev, test, opt.shuffle)
            elif dataset == Data.sigmorphon16task1:
                if opt.indtag:
                    self.data = dataloader.TagSIGMORPHON2016Task1(train, dev, test, opt.shuffle)
                else:
                    self.data = dataloader.SIGMORPHON2016Task1(train, dev, test, opt.shuffle)
            elif dataset == Data.lemma:
                if opt.indtag:
                    self.data = dataloader.TagLemmatization(train, dev, test, opt.shuffle)
                else:
                    self.data = dataloader.Lemmatization(train, dev, test, opt.shuffle)
            elif dataset == Data.lemmanotag:
                self.data = dataloader.LemmatizationNotag(train, dev, test, opt.shuffle)
            else:
                raise ValueError
        # yapf: enable
        logger.info('src vocab size %d', self.data.source_vocab_size)
        logger.info('trg vocab size %d', self.data.target_vocab_size)
        logger.info('src vocab %r', self.data.source[:500])
        logger.info('trg vocab %r', self.data.target[:500])

    def build_model(self, opt):
        assert self.model is None
        if opt.arch == Arch.hardmono:
            opt.indtag, opt.mono = True, True
        params = dict()
        params['src_vocab_size'] = self.data.source_vocab_size
        params['trg_vocab_size'] = self.data.target_vocab_size
        params['embed_dim'] = opt.embed_dim
        params['dropout_p'] = opt.dropout
        params['src_hid_size'] = opt.src_hs
        params['trg_hid_size'] = opt.trg_hs
        params['src_nb_layers'] = opt.src_layer
        params['trg_nb_layers'] = opt.trg_layer
        params['nb_attr'] = self.data.nb_attr
        params['nb_sample'] = opt.nb_sample
        params['wid_siz'] = opt.wid_siz
        params['src_c2i'] = self.data.source_c2i
        params['trg_c2i'] = self.data.target_c2i
        params['attr_c2i'] = self.data.attr_c2i
        model_class = None
        indtag, mono = True, True
        # yapf: disable
        fancy_classfactory = {
            (Arch.hardmono, indtag, mono): model.HardMonoTransducer,
            (Arch.soft, indtag, not mono): model.TagTransducer,
            (Arch.hard, indtag, not mono): model.TagHardAttnTransducer,
            (Arch.hmm, indtag, not mono): model.TagHMMTransducer,
            (Arch.hmm, indtag, mono): model.MonoTagHMMTransducer,
            (Arch.hmmfull, indtag, not mono): model.TagFullHMMTransducer,
            (Arch.hmmfull, indtag, mono): model.MonoTagFullHMMTransducer,
        }
        regular_classfactory = {
            Arch.soft: model.Transducer,
            Arch.hard: model.HardAttnTransducer,
            Arch.softinputfeed: model.InputFeedTransducer,
            Arch.largesoftinputfeed: model.LargeInputFeedTransducer,
            Arch.approxihard: model.ApproxiHardTransducer,
            Arch.approxihardinputfeed: model.ApproxiHardInputFeedTransducer,
            Arch.hmm: model.HMMTransducer,
            Arch.hmmfull: model.FullHMMTransducer
        }
        # yapf: enable
        if opt.indtag or opt.mono:
            model_class = fancy_classfactory[(opt.arch, opt.copy, opt.indtag,
                                              opt.mono)]
        else:
            model_class = regular_classfactory[opt.arch]
        self.model = model_class(**params)
        if opt.indtag:
            self.logger.info('number of attribute %d', self.model.nb_attr)
            self.logger.info('dec 1st rnn %r', self.model.dec_rnn.layers[0])
        if opt.arch in [
                Arch.softinputfeed, Arch.approxihardinputfeed,
                Arch.largesoftinputfeed
        ]:
            self.logger.info('merge_input with %r', self.model.merge_input)
        self.logger.info('number of parameter %d',
                         self.model.count_nb_params())
        self.model = self.model.to(self.device)

    def dump_state_dict(self, filepath):
        util.maybe_mkdir(filepath)
        self.model = self.model.to('cpu')
        torch.save(self.model.state_dict(), filepath)
        self.model = self.model.to(self.device)
        self.logger.info(f'dump to {filepath}')

    def load_state_dict(self, filepath):
        state_dict = torch.load(filepath)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.logger.info(f'load from {filepath}')

    def setup_evalutator(self, arch, dataset):
        if arch == Arch.hardmono:
            if dataset == Data.news15:
                self.evaluator = util.PairTranslitEvaluator()
            elif dataset == Data.sigmorphon17task1:
                self.evaluator = util.PairBasicEvaluator()
            elif dataset == Data.g2p:
                self.evaluator = util.PairG2PEvaluator()
            else:
                raise ValueError
        else:
            if dataset == Data.news15:
                self.evaluator = util.TranslitEvaluator()
            elif dataset == Data.g2p:
                self.evaluator = util.G2PEvaluator()
            elif dataset == Data.p2g:
                self.evaluator = util.P2GEvaluator()
            else:
                self.evaluator = util.BasicEvaluator()

    def evaluate(self, mode, epoch_idx, decode_fn):
        self.model.eval()
        sampler, nb_instance = self.iterate_instance(mode)
        results = self.evaluator.evaluate_all(sampler, nb_instance, self.model,
                                              decode_fn)
        for result in results:
            self.logger.info(
                f'{mode} {result.long_desc} is {result.res} at epoch {epoch_idx}'
            )
        return results

    def decode(self, mode, write_fp, decode_fn):
        self.model.eval()
        cnt = 0
        sampler, nb_instance = self.iterate_instance(mode)
        with open(f'{write_fp}.{mode}.tsv', 'w') as fp:
            fp.write(f'prediction\ttarget\tloss\tdist\n')
            for src, trg in tqdm(sampler(), total=nb_instance):
                pred, _ = decode_fn(self.model, src)
                dist = util.edit_distance(pred, trg.view(-1).tolist()[1:-1])

                src_mask = dummy_mask(src)
                out = self.model(src, src_mask, trg)
                loss = self.model.loss(out, trg[1:]).item()

                trg = self.data.decode_target(trg)[1:-1]
                pred = self.data.decode_target(pred)
                fp.write(
                    f'{" ".join(pred)}\t{" ".join(trg)}\t{loss}\t{dist}\n')
                cnt += 1
        self.logger.info(f'finished decoding {cnt} {mode} instance')

    def select_model(self, opt):
        best_fp, _, best_res = self.models[0]
        best_acc_fp, _, best_acc = self.models[0]
        best_devloss_fp, best_devloss, _ = self.models[0]
        for fp, devloss, res in self.models:
            if type(self.evaluator) == util.BasicEvaluator or \
               type(self.evaluator) == util.G2PEvaluator or \
               type(self.evaluator) == util.P2GEvaluator:
                # [acc, edit distance / per ]
                if res[0].res >= best_res[0].res and \
                   res[1].res <= best_res[1].res:
                    best_fp, best_res = fp, res
            elif type(self.evaluator) == util.TranslitEvaluator:
                if res[0].res >= best_res[0].res and \
                   res[1].res >= best_res[1].res:
                    best_fp, best_res = fp, res
            else:
                raise NotImplementedError
            if res[0].res >= best_acc[0].res:
                best_acc_fp, best_acc = fp, res
            if devloss <= best_devloss:
                best_devloss_fp, best_devloss = fp, devloss
        if opt.bestacc:
            best_fp = best_acc_fp
        return best_fp, set([best_fp])


def main():
    '''
    main
    '''
    opt = get_args()
    util.maybe_mkdir(opt.model)
    logger = util.get_logger(opt.model + '.log', log_level=opt.loglevel)
    if opt.dataset != Data.sigmorphon19task1:
        opt.train = opt.train[0]
    for key, value in vars(opt).items():
        logger.info('command line argument: %s - %r', key, value)
    setup_seed(opt.seed)

    trainer = Trainer(logger)
    decode_fn = get_decode_fn(opt.decode, opt.max_decode_len)
    trainer.load_data(opt.dataset, opt.train, opt.dev, opt.test, opt)
    trainer.setup_evalutator(opt.arch, opt.dataset)
    if opt.load and opt.load != '0':
        if opt.load == 'smart':
            start_epoch = trainer.smart_load_model(opt.model) + 1
        else:
            start_epoch = trainer.load_model(opt.load) + 1
        logger.info('continue training from epoch %d', start_epoch)
        trainer.setup_training(opt.optimizer, opt.lr, opt.momentum)
        trainer.setup_scheduler(opt.min_lr, opt.patience, opt.cooldown,
                                opt.discount_factor)
        trainer.load_training(opt.model)
    else:  # start from scratch
        start_epoch = 0
        trainer.build_model(opt)
        if opt.init:
            if os.path.isfile(opt.init):
                trainer.load_state_dict(opt.init)
            else:
                trainer.dump_state_dict(opt.init)
        trainer.setup_training(opt.optimizer, opt.lr, opt.momentum)
        trainer.setup_scheduler(opt.min_lr, opt.patience, opt.cooldown,
                                opt.discount_factor)

    trainer.run(opt, start_epoch, decode_fn=decode_fn)


if __name__ == '__main__':
    main()
