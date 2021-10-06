from collections import namedtuple

import torch

import util
from dataloader import BOS_IDX, EOS_IDX, STEP_IDX
from model import HardMonoTransducer, HMMTransducer, dummy_mask
from transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decode(util.NamedEnum):
    greedy = "greedy"
    beam = "beam"


class Decoder(object):
    def __init__(
        self,
        decoder_type,
        max_len=100,
        beam_size=5,
        trg_bos=BOS_IDX,
        trg_eos=EOS_IDX,
        skip_attn=True,
    ):
        self.type = decoder_type
        self.max_len = max_len
        self.beam_size = beam_size
        self.trg_bos = trg_bos
        self.trg_eos = trg_eos
        self.skip_attn = skip_attn

    def __call__(self, transducer, src_sentence, src_mask):
        if self.type == Decode.greedy:
            if isinstance(transducer, HardMonoTransducer):
                decode_fn = decode_greedy_mono
            elif isinstance(transducer, HMMTransducer):
                decode_fn = decode_greedy_hmm
            elif isinstance(transducer, Transformer):
                decode_fn = decode_greedy_transformer
            else:
                decode_fn = decode_greedy_default

            output, attns = decode_fn(
                transducer,
                src_sentence,
                src_mask,
                max_len=self.max_len,
                trg_bos=self.trg_bos,
                trg_eos=self.trg_eos,
            )
        elif self.type == Decode.beam:
            if isinstance(transducer, HardMonoTransducer):
                decode_fn = decode_beam_mono
            elif isinstance(transducer, HMMTransducer):
                decode_fn = decode_beam_hmm
            elif isinstance(transducer, Transformer):
                decode_fn = decode_beam_transformer
            else:
                decode_fn = decode_beam_search_default

            output, attns = decode_fn(
                transducer,
                src_sentence,
                src_mask,
                max_len=self.max_len,
                nb_beam=self.beam_size,
                trg_bos=self.trg_bos,
                trg_eos=self.trg_eos,
            )
        else:
            raise ValueError
        return_values = (output, None if self.skip_attn else attns)
        return return_values


def get_decode_fn(decode, max_len=100, beam_size=5):
    return Decoder(decode, max_len=max_len, beam_size=beam_size)


def decode_greedy_default(
    transducer, src_sentence, src_mask, max_len=100, trg_bos=BOS_IDX, trg_eos=EOS_IDX
):
    """
    src_sentence: [seq_len]
    """
    transducer.eval()
    enc_hs = transducer.encode(src_sentence)
    _, bs = src_mask.shape

    hidden = transducer.dec_rnn.get_init_hx(bs)
    input_ = torch.tensor([trg_bos] * bs, device=DEVICE)
    output = input_.view(1, bs)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    attns = []

    finished = None
    for _ in range(max_len):
        word_logprob, hidden, attn = transducer.decode_step(
            enc_hs, src_mask, input_, hidden
        )
        word = torch.max(word_logprob, dim=1)[1]
        attns.append(attn)

        input_ = transducer.dropout(transducer.trg_embed(word))
        output = torch.cat((output, word.view(1, bs)))

        if finished is None:
            finished = word == trg_eos
        else:
            finished = finished | (word == trg_eos)

        if finished.all().item():
            break

    return output, attns


def decode_greedy_mono(
    transducer, src_sentence, src_mask, max_len=100, trg_bos=BOS_IDX, trg_eos=EOS_IDX
):
    """
    src_sentence: [seq_len]
    """
    assert isinstance(transducer, HardMonoTransducer)
    transducer.eval()
    enc_hs = transducer.encode(src_sentence)
    seq_len, bs = src_mask.shape

    attn_pos = torch.tensor([0] * bs, device=DEVICE).view(1, -1)
    hidden = transducer.dec_rnn.get_init_hx(bs)
    input_ = torch.tensor([trg_bos] * bs, device=DEVICE)
    output = input_.view(1, bs)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    attns = []

    finished = None
    for _ in range(max_len):
        word_logprob, hidden, attn = transducer.decode_step(
            enc_hs, src_mask, input_, hidden, attn_pos
        )
        word = torch.max(word_logprob, dim=1)[1]
        attns.append(attn)

        input_ = transducer.dropout(transducer.trg_embed(word))
        output = torch.cat((output, word.view(1, bs)))

        attn_pos = attn_pos + (word == STEP_IDX)
        attn_pos = attn_pos.clamp_max(seq_len - 1)

        if finished is None:
            finished = word == trg_eos
        else:
            finished = finished | (word == trg_eos)

        if finished.all().item():
            break

    return output, attns


def decode_greedy_hmm(
    transducer, src_sentence, src_mask, max_len=100, trg_bos=BOS_IDX, trg_eos=EOS_IDX
):
    transducer.eval()
    enc_hs = transducer.encode(src_sentence)
    T, bs = src_mask.shape

    hidden = transducer.dec_rnn.get_init_hx(bs)
    input_ = torch.tensor([trg_bos] * bs, device=DEVICE)
    output = input_.view(1, bs)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    attns = []

    finished = None
    for idx in range(max_len):
        trans, emiss, hidden = transducer.decode_step(enc_hs, src_mask, input_, hidden)
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

        input_ = transducer.dropout(transducer.trg_embed(word))
        output = torch.cat((output, word.view(1, bs)))

        if finished is None:
            finished = word == trg_eos
        else:
            finished = finished | (word == trg_eos)

        if finished.all().item():
            break

        word_idx = word.view(-1, 1).expand(bs, T).unsqueeze(-1)
        word_emiss = torch.gather(emiss, -1, word_idx).view(bs, 1, T)
        forward = forward + word_emiss
    return output, attns


def decode_greedy_transformer(
    transducer, src_sentence, src_mask, max_len=100, trg_bos=BOS_IDX, trg_eos=EOS_IDX
):
    """
    src_sentence: [seq_len]
    """
    assert isinstance(transducer, Transformer)
    transducer.eval()
    src_mask = (src_mask == 0).transpose(0, 1)
    enc_hs = transducer.encode(src_sentence, src_mask)

    _, bs = src_sentence.shape
    output = torch.tensor([trg_bos] * bs, device=DEVICE)
    output = output.view(1, bs)

    finished = None
    for _ in range(max_len):
        trg_mask = dummy_mask(output)
        trg_mask = (trg_mask == 0).transpose(0, 1)

        word_logprob = transducer.decode(enc_hs, src_mask, output, trg_mask)
        word_logprob = word_logprob[-1]

        word = torch.max(word_logprob, dim=1)[1]
        output = torch.cat((output, word.view(1, bs)))

        if finished is None:
            finished = word == trg_eos
        else:
            finished = finished | (word == trg_eos)

        if finished.all().item():
            break
    return output, None


Beam = namedtuple("Beam", "log_prob hidden input partial_sent")


def get_topk_beam_idx(next_beams, nb_beam):
    return torch.stack([b.log_prob for b in next_beams], dim=-1).topk(nb_beam).indices


def gather_logprob(next_beams, bs, beam_idx):
    return torch.stack([b.log_prob for b in next_beams], dim=-1)[
        torch.arange(bs).view(-1, 1), beam_idx
    ]


def gather_lstm_hidden(next_beams, bs, beam_idx, hidden_idx=0):
    nb_layer, _, hid_dim = next_beams[0].hidden[hidden_idx].shape
    return torch.stack([b.hidden[hidden_idx] for b in next_beams], dim=-1)[
        torch.arange(nb_layer).view(-1, 1, 1, 1),
        torch.arange(bs).view(1, -1, 1, 1),
        torch.arange(hid_dim).view(1, 1, -1, 1),
        beam_idx.unsqueeze(1).unsqueeze(0),
    ]


def gather_lstm_input(next_beams, bs, emb_dim, beam_idx):
    return torch.stack([b.input for b in next_beams], dim=-1)[
        torch.arange(bs).view(-1, 1, 1),
        torch.arange(emb_dim).view(1, -1, 1),
        beam_idx.unsqueeze(1),
    ]


def gather_output(next_beams, cur_len, bs, beam_idx):
    return torch.stack([b.partial_sent for b in next_beams], dim=-1)[
        torch.arange(cur_len).view(-1, 1, 1),
        torch.arange(bs).view(1, -1, 1),
        beam_idx,
    ]


def decode_beam_search_default(
    transducer,
    src_sentence,
    src_mask,
    max_len=50,
    nb_beam=5,
    trg_bos=BOS_IDX,
    trg_eos=EOS_IDX,
):
    """
    src_sentence: [seq_len]
    """

    transducer.eval()
    enc_hs = transducer.encode(src_sentence)
    _, bs = src_mask.shape

    hidden = transducer.dec_rnn.get_init_hx(bs)
    input_ = torch.tensor([trg_bos] * bs, device=DEVICE)
    output = input_.view(1, bs)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    _, emb_dim = input_.shape
    start = Beam(0, hidden, input_, output)
    beams = [start]
    finish_beams = [list() for _ in range(bs)]
    for i in range(max_len):
        cur_len = i + 2  # bos & the current prediction
        next_beams = []
        for beam in beams:
            word_logprob, hidden, _ = transducer.decode_step(
                enc_hs, src_mask, beam.input, beam.hidden
            )
            topk_log_prob, topk_word = word_logprob.topk(nb_beam)
            topk_log_prob = topk_log_prob.split(1, dim=1)
            topk_word = topk_word.split(1, dim=1)
            for log_prob, word in zip(topk_log_prob, topk_word):
                log_prob = log_prob.squeeze(1)
                word = word.squeeze(1)
                log_prob = beam.log_prob + log_prob
                input_ = transducer.dropout(transducer.trg_embed(word))
                output = torch.cat((beam.partial_sent, word.view(1, bs)))

                if (word == trg_eos).any():
                    batch_idx = (word == trg_eos).nonzero().view(-1).tolist()
                    for j in batch_idx:
                        score = log_prob[j] / cur_len
                        seq = output[:, j].tolist()
                        log_prob[j] = -1e6
                        finish_beams[j].append((score, seq))

                new_beam = Beam(
                    log_prob,
                    hidden,
                    input_,
                    output,
                )
                next_beams.append(new_beam)

        beam_idx = get_topk_beam_idx(next_beams, nb_beam)
        log_prob = gather_logprob(next_beams, bs, beam_idx)
        hidden0 = gather_lstm_hidden(next_beams, bs, beam_idx, hidden_idx=0)
        hidden1 = gather_lstm_hidden(next_beams, bs, beam_idx, hidden_idx=1)
        input_ = gather_lstm_input(next_beams, bs, emb_dim, beam_idx)
        output = gather_output(next_beams, cur_len, bs, beam_idx)

        beams = [
            Beam(
                lp.squeeze(-1),
                (h0.squeeze(-1), h1.squeeze(-1)),
                ip.squeeze(-1),
                op.squeeze(-1),
            )
            for lp, h0, h1, ip, op in zip(
                log_prob.split(1, dim=-1),
                hidden0.split(1, dim=-1),
                hidden1.split(1, dim=-1),
                input_.split(1, dim=-1),
                output.split(1, dim=-1),
            )
        ]
    return [max(b)[1] if b else [] for b in finish_beams], None


def decode_beam_transformer(
    transducer,
    src_sentence,
    src_mask,
    max_len=50,
    nb_beam=5,
    trg_bos=BOS_IDX,
    trg_eos=EOS_IDX,
):
    """
    src_sentence: [seq_len]
    """
    assert isinstance(transducer, Transformer)

    transducer.eval()
    src_mask = (src_mask == 0).transpose(0, 1)
    enc_hs = transducer.encode(src_sentence, src_mask)

    _, bs = src_sentence.shape
    input_ = torch.tensor([trg_bos] * bs, device=DEVICE)
    input_ = input_.view(1, bs)
    output = input_
    start = Beam(0, None, input_, output)
    beams = [start]
    finish_beams = [list() for _ in range(bs)]
    for i in range(max_len):
        cur_len = i + 2  # bos & the current prediction
        next_beams = []
        for beam in beams:
            trg_mask = dummy_mask(beam.input)
            trg_mask = (trg_mask == 0).transpose(0, 1)

            word_logprob = transducer.decode(enc_hs, src_mask, beam.input, trg_mask)
            word_logprob = word_logprob[-1]

            topk_log_prob, topk_word = word_logprob.topk(nb_beam)
            topk_log_prob = topk_log_prob.split(1, dim=1)
            topk_word = topk_word.split(1, dim=1)
            for log_prob, word in zip(topk_log_prob, topk_word):
                log_prob = log_prob.squeeze(1)
                word = word.squeeze(1)

                log_prob = beam.log_prob + log_prob
                input_ = torch.cat((beam.input, word.view(1, bs)))
                output = torch.cat((beam.partial_sent, word.view(1, bs)))

                if (word == trg_eos).any():
                    batch_idx = (word == trg_eos).nonzero().view(-1).tolist()
                    for j in batch_idx:
                        score = log_prob[j] / cur_len
                        seq = output[:, j].tolist()
                        log_prob[j] = -1e6
                        finish_beams[j].append((score, seq))

                new_beam = Beam(
                    log_prob,
                    None,
                    input_,
                    output,
                )
                next_beams.append(new_beam)

        beam_idx = get_topk_beam_idx(next_beams, nb_beam)
        log_prob = gather_logprob(next_beams, bs, beam_idx)
        input_ = torch.stack([b.input for b in next_beams], dim=-1)[
            torch.arange(cur_len).view(-1, 1, 1),
            torch.arange(bs).view(1, -1, 1),
            beam_idx,
        ]
        output = gather_output(next_beams, cur_len, bs, beam_idx)
        beams = [
            Beam(lp.squeeze(-1), None, ip.squeeze(-1), op.squeeze(-1))
            for lp, ip, op in zip(
                log_prob.split(1, dim=-1),
                input_.split(1, dim=-1),
                output.split(1, dim=-1),
            )
        ]
    return [max(b)[1] if b else [] for b in finish_beams], None


BeamHard = namedtuple("BeamHard", "log_prob hidden input partial_sent attn_pos")


def decode_beam_mono(
    transducer,
    src_sentence,
    src_mask,
    max_len=50,
    nb_beam=5,
    trg_bos=BOS_IDX,
    trg_eos=EOS_IDX,
):
    assert isinstance(transducer, HardMonoTransducer)

    transducer.eval()
    enc_hs = transducer.encode(src_sentence)
    seq_len, bs = src_mask.shape

    attn_pos = torch.tensor([0] * bs, device=DEVICE).view(1, -1)
    hidden = transducer.dec_rnn.get_init_hx(bs)
    input_ = torch.tensor([trg_bos] * bs, device=DEVICE)
    output = input_.view(1, bs)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    _, emb_dim = input_.shape
    start = BeamHard(0, hidden, input_, output, attn_pos)
    beams = [start]
    finish_beams = [list() for _ in range(bs)]
    for i in range(max_len):
        cur_len = i + 2  # bos & the current prediction
        next_beams = []
        for beam in beams:
            word_logprob, hidden, _ = transducer.decode_step(
                enc_hs, src_mask, beam.input, beam.hidden, beam.attn_pos
            )
            topk_log_prob, topk_word = word_logprob.topk(nb_beam)
            topk_log_prob = topk_log_prob.split(1, dim=1)
            topk_word = topk_word.split(1, dim=1)
            for log_prob, word in zip(topk_log_prob, topk_word):
                log_prob = log_prob.squeeze(1)
                word = word.squeeze(1)
                log_prob = beam.log_prob + log_prob
                input_ = transducer.dropout(transducer.trg_embed(word))
                output = torch.cat((beam.partial_sent, word.view(1, bs)))

                attn_pos = beam.attn_pos + (word == STEP_IDX)
                attn_pos = attn_pos.clamp_max(seq_len - 1)

                if (word == trg_eos).any():
                    batch_idx = (word == trg_eos).nonzero().view(-1).tolist()
                    for j in batch_idx:
                        score = log_prob[j] / cur_len
                        seq = output[:, j].tolist()
                        log_prob[j] = -1e6
                        finish_beams[j].append((score, seq))

                new_beam = BeamHard(log_prob, hidden, input_, output, attn_pos)
                next_beams.append(new_beam)

        beam_idx = get_topk_beam_idx(next_beams, nb_beam)
        log_prob = gather_logprob(next_beams, bs, beam_idx)
        hidden0 = gather_lstm_hidden(next_beams, bs, beam_idx, hidden_idx=0)
        hidden1 = gather_lstm_hidden(next_beams, bs, beam_idx, hidden_idx=1)
        input_ = gather_lstm_input(next_beams, bs, emb_dim, beam_idx)
        output = gather_output(next_beams, cur_len, bs, beam_idx)
        attn_pos = torch.stack([b.attn_pos for b in next_beams], dim=-1)[
            0, torch.arange(bs).view(1, -1, 1), beam_idx.unsqueeze(0)
        ]
        beams = [
            BeamHard(
                lp.squeeze(-1),
                (h0.squeeze(-1), h1.squeeze(-1)),
                ip.squeeze(-1),
                op.squeeze(-1),
                ap.squeeze(-1),
            )
            for lp, h0, h1, ip, op, ap in zip(
                log_prob.split(1, dim=-1),
                hidden0.split(1, dim=-1),
                hidden1.split(1, dim=-1),
                input_.split(1, dim=-1),
                output.split(1, dim=-1),
                attn_pos.split(1, dim=-1),
            )
        ]
    return [max(b)[1] if b else [] for b in finish_beams], None


BeamHMM = namedtuple("BeamHMM", "log_prob hidden input partial_sent forward")


def decode_beam_hmm(
    transducer,
    src_sentence,
    src_mask,
    max_len=50,
    nb_beam=5,
    trg_bos=BOS_IDX,
    trg_eos=EOS_IDX,
    return_top_beams=False,
):

    transducer.eval()
    enc_hs = transducer.encode(src_sentence)
    T, bs = src_mask.shape

    hidden = transducer.dec_rnn.get_init_hx(bs)
    input_ = torch.tensor([trg_bos] * bs, device=DEVICE)
    output = input_.view(1, bs)
    input_ = transducer.dropout(transducer.trg_embed(input_))
    _, emb_dim = input_.shape
    forward = None
    start = BeamHMM(0, hidden, input_, output, forward)
    beams = [start]
    finish_beams = [list() for _ in range(bs)]
    for i in range(max_len):
        cur_len = i + 2  # bos & the current prediction
        next_beams = []
        for beam in beams:
            trans, emiss, hidden = transducer.decode_step(
                enc_hs, src_mask, beam.input, beam.hidden
            )

            if beam.forward is None:
                initial = trans[:, 0].unsqueeze(1)
                forward = initial
            else:
                # forward = torch.bmm(forward, trans)
                forward = beam.forward + trans.transpose(1, 2)
                forward = forward.logsumexp(dim=-1, keepdim=True).transpose(1, 2)

            # wordprob = torch.bmm(forward, emiss)
            log_wordprob = forward + emiss.transpose(1, 2)
            log_wordprob = log_wordprob.logsumexp(dim=-1)
            topk_word = torch.topk(log_wordprob, nb_beam, dim=-1)[1]
            for word in topk_word.split(1, dim=1):
                word = word.squeeze(1)
                input_ = transducer.dropout(transducer.trg_embed(word))
                output = torch.cat((beam.partial_sent, word.view(1, bs)))

                word_idx = word.view(-1, 1).expand(bs, T).unsqueeze(-1)
                word_emiss = torch.gather(emiss, -1, word_idx).view(bs, 1, T)
                next_forward = forward + word_emiss

                log_prob = torch.logsumexp(next_forward, dim=-1).squeeze(1)

                if (word == trg_eos).any():
                    batch_idx = (word == trg_eos).nonzero().view(-1).tolist()
                    for j in batch_idx:
                        score = log_prob[j] / cur_len
                        seq = output[:, j].tolist()
                        log_prob[j] = -1e6
                        finish_beams[j].append((score, seq))

                new_beam = BeamHMM(
                    log_prob,
                    hidden,
                    input_,
                    output,
                    next_forward,
                )
                next_beams.append(new_beam)

        beam_idx = get_topk_beam_idx(next_beams, nb_beam)
        log_prob = gather_logprob(next_beams, bs, beam_idx)
        hidden0 = gather_lstm_hidden(next_beams, bs, beam_idx, hidden_idx=0)
        hidden1 = gather_lstm_hidden(next_beams, bs, beam_idx, hidden_idx=1)
        input_ = gather_lstm_input(next_beams, bs, emb_dim, beam_idx)
        output = gather_output(next_beams, cur_len, bs, beam_idx)
        forward = torch.stack([b.forward for b in next_beams], dim=-1)[
            torch.arange(bs).view(-1, 1, 1, 1),
            0,
            torch.arange(T).view(1, 1, -1, 1),
            beam_idx.unsqueeze(1).unsqueeze(1),
        ]

        beams = [
            BeamHMM(
                lp.squeeze(-1),
                (h0.squeeze(-1), h1.squeeze(-1)),
                ip.squeeze(-1),
                op.squeeze(-1),
                fw.squeeze(-1),
            )
            for lp, h0, h1, ip, op, fw in zip(
                log_prob.split(1, dim=-1),
                hidden0.split(1, dim=-1),
                hidden1.split(1, dim=-1),
                input_.split(1, dim=-1),
                output.split(1, dim=-1),
                forward.split(1, dim=-1),
            )
        ]
    return [max(b)[1] if b else [] for b in finish_beams], None
