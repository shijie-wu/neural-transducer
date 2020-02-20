from collections import namedtuple
from functools import partial

import torch

import util
from dataloader import BOS_IDX, EOS_IDX, STEP_IDX
from model import Categorical, HardMonoTransducer, HMMTransducer, dummy_mask
from transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decode(util.NamedEnum):
    greedy = 'greedy'
    sample = 'sample'
    beam = 'beam'


class Decoder(object):
    def __init__(self,
                 decoder_type,
                 max_len=100,
                 trg_bos=BOS_IDX,
                 trg_eos=EOS_IDX,
                 skip_attn=True):
        self.type = decoder_type
        self.max_len = max_len
        self.trg_bos = trg_bos
        self.trg_eos = trg_eos
        self.skip_attn = skip_attn
        self.cache = {}

    def reset(self):
        self.cache = {}

    def src2str(self, src_sentence):
        def tensor2str(tensor):
            return str(tensor.view(-1).cpu().numpy())

        if isinstance(src_sentence, tuple) and all([isinstance(x, torch.Tensor) for x in src_sentence]):
            return str([tensor2str(x) for x in src_sentence])
        elif isinstance(src_sentence, torch.Tensor):
            return tensor2str(src_sentence)
        else:
            raise ValueError(src_sentence)


    def __call__(self, transducer, src_sentence):
        key = self.src2str(src_sentence)
        if key in self.cache:
            return self.cache[key]
        if self.type == Decode.greedy:
            output, attns = decode_greedy(transducer,
                                          src_sentence,
                                          max_len=self.max_len,
                                          trg_bos=self.trg_bos,
                                          trg_eos=self.trg_eos)
        elif self.type == Decode.beam:
            output, attns = decode_beam_search(transducer,
                                               src_sentence,
                                               max_len=self.max_len,
                                               trg_bos=self.trg_bos,
                                               trg_eos=self.trg_eos)
        elif self.type == Decode.sample:
            output, attns = decode_sample(transducer,
                                          src_sentence,
                                          max_len=self.max_len,
                                          trg_bos=self.trg_bos,
                                          trg_eos=self.trg_eos)
        else:
            raise ValueError
        return_values = (output, None if self.skip_attn else attns)
        # don't cache sampling results
        if self.type == Decode.sample:
            return return_values
        else:
            self.cache[key] = return_values
            return self.cache[key]


def get_decode_fn(decode, max_len=100):
    return Decoder(decode, max_len=max_len)


def decode_sample(transducer,
                  src_sentence,
                  max_len=100,
                  trg_bos=BOS_IDX,
                  trg_eos=EOS_IDX):
    '''
    src_sentence: [seq_len]
    '''
    assert not isinstance(transducer, HardMonoTransducer)
    if isinstance(transducer, HMMTransducer):
        return decode_sample_hmm(transducer,
                                 src_sentence,
                                 max_len=max_len,
                                 trg_bos=BOS_IDX,
                                 trg_eos=EOS_IDX)
    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)

    output, attns = [], []
    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    for _ in range(max_len):
        word_logprob, hidden, attn = transducer.decode_step(
            enc_hs, src_mask, input_, hidden)
        word = Categorical(word_logprob.exp()).sample_n(1)[0]
        attns.append(attn)
        if word == trg_eos:
            break
        input_ = transducer.dropout(transducer.trg_embed(word))
        output.append(word.item())
    return output, attns


def decode_sample_hmm(transducer,
                      src_sentence,
                      max_len=100,
                      trg_bos=BOS_IDX,
                      trg_eos=EOS_IDX):
    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)
    T = src_mask.shape[0]

    output, attns = [], []
    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    for idx in range(max_len):
        trans, emiss, hidden = transducer.decode_step(enc_hs, src_mask, input_,
                                                      hidden)
        if idx == 0:
            initial = trans[:, 0].unsqueeze(1)
            attns.append(initial)
            forward = initial
        else:
            attns.append(trans)
            # forward = torch.bmm(forward, trans)
            forward = forward + trans.transpose(1, 2)
            forward = forward.logsumexp(dim=-1, keepdim=True).transpose(1, 2)

        # wordprob = torch.bmm(forward, emiss)
        log_wordprob = forward + emiss.transpose(1, 2)
        log_wordprob = log_wordprob.logsumexp(dim=-1)
        # word = torch.max(log_wordprob, dim=-1)[1]
        word = Categorical(log_wordprob.exp()).sample_n(1)[0]
        if word == trg_eos:
            break
        input_ = transducer.dropout(transducer.trg_embed(word))
        output.append(word.item())
        word_idx = word.view(-1, 1).expand(1, T).unsqueeze(-1)
        word_emiss = torch.gather(emiss, -1, word_idx).view(1, 1, T)
        forward = forward + word_emiss
    return output, attns


def decode_greedy(transducer,
                  src_sentence,
                  max_len=100,
                  trg_bos=BOS_IDX,
                  trg_eos=EOS_IDX):
    '''
    src_sentence: [seq_len]
    '''
    if isinstance(transducer, HardMonoTransducer):
        return decode_greedy_mono(transducer,
                                  src_sentence,
                                  max_len=max_len,
                                  trg_bos=BOS_IDX,
                                  trg_eos=EOS_IDX)
    if isinstance(transducer, HMMTransducer):
        return decode_greedy_hmm(transducer,
                                 src_sentence,
                                 max_len=max_len,
                                 trg_bos=BOS_IDX,
                                 trg_eos=EOS_IDX)
    if isinstance(transducer, Transformer):
        return decode_greedy_transformer(transducer,
                                         src_sentence,
                                         max_len=max_len,
                                         trg_bos=BOS_IDX,
                                         trg_eos=EOS_IDX)
    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)

    output, attns = [], []
    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    for _ in range(max_len):
        word_logprob, hidden, attn = transducer.decode_step(
            enc_hs, src_mask, input_, hidden)
        word = torch.max(word_logprob, dim=1)[1]
        attns.append(attn)
        if word == trg_eos:
            break
        input_ = transducer.dropout(transducer.trg_embed(word))
        output.append(word.item())
    return output, attns


def decode_greedy_mono(transducer,
                       src_sentence,
                       max_len=100,
                       trg_bos=BOS_IDX,
                       trg_eos=EOS_IDX):
    '''
    src_sentence: [seq_len]
    '''
    assert isinstance(transducer, HardMonoTransducer)
    attn_pos = 0
    transducer.eval()
    if isinstance(src_sentence, tuple):
        seq_len = src_sentence[0].shape[0]
    else:
        seq_len = src_sentence.shape[0]
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)

    output, attns = [], []
    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    for _ in range(max_len):
        word_logprob, hidden, attn = transducer.decode_step(
            enc_hs, src_mask, input_, hidden, attn_pos)
        word = torch.max(word_logprob, dim=1)[1]
        attns.append(attn)
        if word == STEP_IDX:
            attn_pos += 1
            if attn_pos == seq_len:
                attn_pos = seq_len - 1
        if word == trg_eos:
            break
        input_ = transducer.dropout(transducer.trg_embed(word))
        output.append(word.item())
    return output, attns


def decode_greedy_hmm(transducer,
                      src_sentence,
                      max_len=100,
                      trg_bos=BOS_IDX,
                      trg_eos=EOS_IDX):
    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)
    T = src_mask.shape[0]

    output, attns = [], []
    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    for idx in range(max_len):
        trans, emiss, hidden = transducer.decode_step(enc_hs, src_mask, input_,
                                                      hidden)
        if idx == 0:
            initial = trans[:, 0].unsqueeze(1)
            attns.append(initial)
            forward = initial
        else:
            attns.append(trans)
            # forward = torch.bmm(forward, trans)
            forward = forward + trans.transpose(1, 2)
            forward = forward.logsumexp(dim=-1, keepdim=True).transpose(1, 2)

        # wordprob = torch.bmm(forward, emiss)
        log_wordprob = forward + emiss.transpose(1, 2)
        log_wordprob = log_wordprob.logsumexp(dim=-1)
        word = torch.max(log_wordprob, dim=-1)[1]
        if word == trg_eos:
            break
        input_ = transducer.dropout(transducer.trg_embed(word))
        output.append(word.item())
        word_idx = word.view(-1, 1).expand(1, T).unsqueeze(-1)
        word_emiss = torch.gather(emiss, -1, word_idx).view(1, 1, T)
        forward = forward + word_emiss
    return output, attns


def decode_greedy_transformer(transducer,
                              src_sentence,
                              max_len=100,
                              trg_bos=BOS_IDX,
                              trg_eos=EOS_IDX):
    '''
    src_sentence: [seq_len]
    '''
    assert isinstance(transducer, Transformer)
    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    src_mask = (src_mask == 0).transpose(0, 1)
    enc_hs = transducer.encode(src_sentence, src_mask)

    output, attns = [trg_bos], []

    for _ in range(max_len):
        output_tensor = torch.tensor(output,
                                     device=DEVICE).view(len(output), 1)
        trg_mask = dummy_mask(output_tensor)
        trg_mask = (trg_mask == 0).transpose(0, 1)

        word_logprob = transducer.decode(enc_hs, src_mask, output_tensor,
                                         trg_mask)
        word_logprob = word_logprob[-1]

        word = torch.max(word_logprob, dim=1)[1]
        if word == trg_eos:
            break
        output.append(word.item())
    return output[1:], attns


Beam = namedtuple('Beam', 'seq_len log_prob hidden input partial_sent attn')


def decode_beam_search(transducer,
                       src_sentence,
                       max_len=50,
                       nb_beam=5,
                       norm=True,
                       trg_bos=BOS_IDX,
                       trg_eos=EOS_IDX):
    '''
    src_sentence: [seq_len]
    '''

    if isinstance(transducer, HardMonoTransducer):
        return decode_beam_mono(transducer,
                                src_sentence,
                                max_len=max_len,
                                nb_beam=nb_beam,
                                norm=norm,
                                trg_bos=BOS_IDX,
                                trg_eos=EOS_IDX)

    if isinstance(transducer, HardMonoTransducer):
        return decode_beam_hmm(transducer,
                               src_sentence,
                               max_len=max_len,
                               nb_beam=nb_beam,
                               norm=norm,
                               trg_bos=BOS_IDX,
                               trg_eos=EOS_IDX)

    def score(beam):
        '''
        compute score based on logprob
        '''
        assert isinstance(beam, Beam)
        if norm:
            return -beam.log_prob / beam.seq_len
        return -beam.log_prob

    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)

    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    start = Beam(1, 0, hidden, input_, '', [])
    beams = [start]
    finish_beams = []
    for _ in range(max_len):
        next_beams = []
        for beam in sorted(beams, key=score)[:nb_beam]:
            word_logprob, hidden, attn = transducer.decode_step(
                enc_hs, src_mask, beam.input, beam.hidden)
            topk_log_prob, topk_word = word_logprob.topk(nb_beam)
            topk_log_prob = topk_log_prob.view(nb_beam, 1)
            topk_word = topk_word.view(nb_beam, 1)
            for log_prob, word in zip(topk_log_prob, topk_word):
                if word == trg_eos:
                    beam = Beam(beam.seq_len + 1,
                                beam.log_prob + log_prob.item(), None, None,
                                beam.partial_sent, beam.attn + [attn])
                    finish_beams.append(beam)
                    # if len(finish_beams) == 10*K:
                    # max_output = sorted(finish_beams, key=score)[0]
                    # return list(map(int, max_output.partial_sent.split())), max_output.attn
                else:
                    beam = Beam(
                        beam.seq_len + 1, beam.log_prob + log_prob.item(),
                        hidden, transducer.dropout(transducer.trg_embed(word)),
                        ' '.join([beam.partial_sent,
                                  str(word.item())]), beam.attn + [attn])
                    next_beams.append(beam)
        beams = next_beams
    finish_beams = finish_beams if finish_beams else next_beams
    max_output = sorted(finish_beams, key=score)[0]
    return list(map(int, max_output.partial_sent.split())), max_output.attn


BeamHard = namedtuple(
    'BeamHard', 'seq_len log_prob hidden input partial_sent attn attn_pos')


def decode_beam_mono(transducer,
                     src_sentence,
                     max_len=50,
                     nb_beam=5,
                     norm=True,
                     trg_bos=BOS_IDX,
                     trg_eos=EOS_IDX):
    assert isinstance(transducer, HardMonoTransducer)

    def score(beam):
        '''
        compute score based on logprob
        '''
        assert isinstance(beam, BeamHard)
        if norm:
            return -beam.log_prob / beam.seq_len
        return -beam.log_prob

    transducer.eval()
    if isinstance(src_sentence, tuple):
        seq_len = src_sentence[0].shape[0]
    else:
        seq_len = src_sentence.shape[0]
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)

    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    start = BeamHard(1, 0, hidden, input_, '', [], 0)
    beams = [start]
    finish_beams = []
    for _ in range(max_len):
        next_beams = []
        for beam in sorted(beams, key=score)[:nb_beam]:
            word_logprob, hidden, attn = transducer.decode_step(
                enc_hs, src_mask, beam.input, beam.hidden, beam.attn_pos)
            topk_log_prob, topk_word = word_logprob.topk(nb_beam)
            topk_log_prob = topk_log_prob.view(nb_beam, 1)
            topk_word = topk_word.view(nb_beam, 1)
            for log_prob, word in zip(topk_log_prob, topk_word):
                if word == trg_eos:
                    beam = BeamHard(beam.seq_len + 1,
                                    beam.log_prob + log_prob.item(), None,
                                    None, beam.partial_sent,
                                    beam.attn + [attn], beam.attn_pos)
                    finish_beams.append(beam)
                    # if len(finish_beams) == 10*K:
                    # max_output = sorted(finish_beams, key=score)[0]
                    # return list(map(int, max_output.partial_sent.split())), max_output.attn
                else:
                    shift = 1 if word == STEP_IDX and beam.attn_pos + 1 < seq_len else 0
                    beam = BeamHard(
                        beam.seq_len + 1, beam.log_prob + log_prob.item(),
                        hidden, transducer.dropout(transducer.trg_embed(word)),
                        ' '.join([beam.partial_sent,
                                  str(word.item())]), beam.attn + [attn],
                        beam.attn_pos + shift)
                    next_beams.append(beam)
        beams = next_beams
    finish_beams = finish_beams if finish_beams else next_beams
    max_output = sorted(finish_beams, key=score)[0]
    return list(map(int, max_output.partial_sent.split())), max_output.attn


BeamHMM = namedtuple(
    'BeamHMM', 'seq_len log_prob hidden input partial_sent attn forward')


def decode_beam_hmm(transducer,
                    src_sentence,
                    max_len=50,
                    nb_beam=5,
                    norm=True,
                    trg_bos=BOS_IDX,
                    trg_eos=EOS_IDX,
                    return_top_beams=False):
    def score(beam):
        '''
        compute score based on logprob
        '''
        assert isinstance(beam, BeamHMM)
        if norm:
            return -beam.log_prob / beam.seq_len
        return -beam.log_prob

    transducer.eval()
    src_mask = dummy_mask(src_sentence)
    enc_hs = transducer.encode(src_sentence)
    T = src_mask.shape[0]

    output, attns = [], []
    hidden = transducer.dec_rnn.get_init_hx(1)
    input_ = torch.tensor([trg_bos], device=DEVICE)
    input_ = transducer.dropout(transducer.trg_embed(input_))

    seq_len, log_prob, partial_sent, attn, forward = 1, 0, '', [], None
    beam = BeamHMM(seq_len, log_prob, hidden, input_, partial_sent, attn,
                   forward)
    beams = [beam]
    finish_beams = []

    for _ in range(max_len):
        next_beams = []
        for beam in sorted(beams, key=score)[:nb_beam]:
            trans, emiss, hidden = transducer.decode_step(
                enc_hs, src_mask, beam.input, beam.hidden)

            if beam.seq_len == 1:
                assert beam.forward is None
                initial = trans[:, 0].unsqueeze(1)
                attn = initial
                forward = initial
            else:
                assert beam.forward is not None
                attn = trans
                # forward = torch.bmm(forward, trans)
                forward = beam.forward + trans.transpose(1, 2)
                forward = forward.logsumexp(dim=-1,
                                            keepdim=True).transpose(1, 2)

            seq_len = beam.seq_len + 1
            next_attn = beam.attn + [attn]

            # wordprob = torch.bmm(forward, emiss)
            log_wordprob = forward + emiss.transpose(1, 2)
            log_wordprob = log_wordprob.logsumexp(dim=-1)
            topk_word = torch.topk(log_wordprob, nb_beam, dim=-1)[1]
            for word in topk_word.view(nb_beam, 1):
                next_input = transducer.dropout(transducer.trg_embed(word))
                next_output = str(word.item())
                word_idx = word.view(-1, 1).expand(1, T).unsqueeze(-1)
                word_emiss = torch.gather(emiss, -1, word_idx).view(1, 1, T)
                next_forward = forward + word_emiss

                log_prob = torch.logsumexp(next_forward, dim=-1).item()

                if word == trg_eos:
                    sent = beam.partial_sent
                    beam = BeamHMM(seq_len, log_prob, None, None, sent,
                                   next_attn, next_forward)
                    finish_beams.append(beam)
                else:
                    sent = f'{beam.partial_sent} {next_output}'
                    beam = BeamHMM(seq_len, log_prob, hidden, next_input, sent,
                                   next_attn, next_forward)
                    next_beams.append(beam)
        beams = next_beams
    finish_beams = finish_beams if finish_beams else next_beams
    sorted_beams = sorted(finish_beams, key=score)
    if return_top_beams:
        return sorted_beams[:nb_beam]
    else:
        max_output = sorted_beams[0]
        return list(map(int, max_output.partial_sent.split())), max_output.attn
