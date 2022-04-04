"""
all model
"""
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution

from dataloader import PAD_IDX, STEP_IDX

EPSILON = 1e-7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):
    def forward(self, x):
        return x


class StackedLSTM(nn.Module):
    """
    step-by-step stacked LSTM
    """

    def __init__(self, input_siz, rnn_siz, nb_layers, dropout):
        """
        init
        """
        super().__init__()
        self.nb_layers = nb_layers
        self.rnn_siz = rnn_siz
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(nb_layers):
            self.layers.append(nn.LSTMCell(input_siz, rnn_siz))
            input_siz = rnn_siz

    def get_init_hx(self, batch_size):
        """
        initial h0
        """
        h_0_s, c_0_s = [], []
        for _ in range(self.nb_layers):
            h_0 = torch.zeros((batch_size, self.rnn_siz), device=DEVICE)
            c_0 = torch.zeros((batch_size, self.rnn_siz), device=DEVICE)
            h_0_s.append(h_0)
            c_0_s.append(c_0)
        return (h_0_s, c_0_s)

    def forward(self, input, hidden):
        """
        dropout after all output except the last one
        """
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = self.dropout(h_1_i)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Attention(nn.Module):
    """
    attention with mask
    """

    def forward(self, ht, hs, mask, weighted_ctx=True):
        """
        ht: batch x ht_dim
        hs: (seq_len x batch x hs_dim, seq_len x batch x ht_dim)
        mask: seq_len x batch
        """
        hs, hs_ = hs
        # seq_len, batch, _ = hs.size()
        hs = hs.transpose(0, 1)
        hs_ = hs_.transpose(0, 1)
        # hs: batch x seq_len x hs_dim
        # hs_: batch x seq_len x ht_dim
        # hs_ = self.hs2ht(hs)
        # Alignment/Attention Function
        # batch x ht_dim x 1
        ht = ht.unsqueeze(2)
        # batch x seq_len
        score = torch.bmm(hs_, ht).squeeze(2)
        # attn = F.softmax(score, dim=-1)
        attn = F.softmax(score, dim=-1) * mask.transpose(0, 1) + EPSILON
        attn = attn / attn.sum(-1, keepdim=True)

        # Compute weighted sum of hs by attention.
        # batch x 1 x seq_len
        attn = attn.unsqueeze(1)
        if weighted_ctx:
            # batch x hs_dim
            weight_hs = torch.bmm(attn, hs).squeeze(1)
        else:
            weight_hs = None

        return weight_hs, attn


class Transducer(nn.Module):
    """
    seq2seq with soft attention baseline
    """

    def __init__(
        self,
        *,
        src_vocab_size,
        trg_vocab_size,
        embed_dim,
        src_hid_size,
        src_nb_layers,
        trg_hid_size,
        trg_nb_layers,
        dropout_p,
        src_c2i,
        trg_c2i,
        attr_c2i,
        **kwargs
    ):
        """
        init
        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.embed_dim = embed_dim
        self.src_hid_size = src_hid_size
        self.src_nb_layers = src_nb_layers
        self.trg_hid_size = trg_hid_size
        self.trg_nb_layers = trg_nb_layers
        self.dropout_p = dropout_p
        self.src_c2i, self.trg_c2i, self.attr_c2i = src_c2i, trg_c2i, attr_c2i
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.trg_embed = nn.Embedding(trg_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.enc_rnn = nn.LSTM(
            embed_dim,
            src_hid_size,
            src_nb_layers,
            bidirectional=True,
            dropout=dropout_p,
        )
        self.dec_rnn = StackedLSTM(embed_dim, trg_hid_size, trg_nb_layers, dropout_p)
        self.out_dim = trg_hid_size + src_hid_size * 2
        self.scale_enc_hs = nn.Linear(src_hid_size * 2, trg_hid_size)
        self.attn = Attention()
        self.linear_out = nn.Linear(self.out_dim, self.out_dim)
        self.final_out = nn.Linear(self.out_dim, trg_vocab_size)
        self.dropout = nn.Dropout(dropout_p)

    def encode(self, src_batch):
        """
        encoder
        """
        enc_hs, _ = self.enc_rnn(self.dropout(self.src_embed(src_batch)))
        scale_enc_hs = self.scale_enc_hs(enc_hs)
        return enc_hs, scale_enc_hs

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        """
        decode step
        """
        h_t, hidden = self.dec_rnn(input_, hidden)
        ctx, attn = self.attn(h_t, enc_hs, enc_mask)
        # Concatenate the ht and ctx
        # weight_hs: batch x (hs_dim + ht_dim)
        ctx = torch.cat((ctx, h_t), dim=1)
        # ctx: batch x out_dim
        ctx = self.linear_out(ctx)
        ctx = torch.tanh(ctx)
        word_logprob = F.log_softmax(self.final_out(ctx), dim=-1)
        return word_logprob, hidden, attn

    def decode(self, enc_hs, enc_mask, trg_batch):
        """
        enc_hs: tuple(enc_hs, scale_enc_hs)
        """
        trg_seq_len = trg_batch.size(0)
        trg_bat_siz = trg_batch.size(1)
        trg_embed = self.dropout(self.trg_embed(trg_batch))
        output = []
        hidden = self.dec_rnn.get_init_hx(trg_bat_siz)
        for idx in range(trg_seq_len - 1):
            input_ = trg_embed[idx, :]
            word_logprob, hidden, _ = self.decode_step(enc_hs, enc_mask, input_, hidden)
            output += [word_logprob]
        return torch.stack(output)

    def forward(self, src_batch, src_mask, trg_batch):
        """
        only for training
        """
        # trg_seq_len, batch_size = trg_batch.size()
        enc_hs = self.encode(src_batch)
        # output: [trg_seq_len-1, batch_size, vocab_siz]
        output = self.decode(enc_hs, src_mask, trg_batch)
        return output

    def count_nb_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def loss(self, predict, target, reduction=True):
        """
        compute loss
        """
        predict = predict.view(-1, self.trg_vocab_size)
        if not reduction:
            loss = F.nll_loss(
                predict, target.view(-1), ignore_index=PAD_IDX, reduction="none"
            )
            loss = loss.view(target.shape)
            loss = loss.sum(dim=0) / (target != PAD_IDX).sum(dim=0)
            return loss

        return F.nll_loss(predict, target.view(-1), ignore_index=PAD_IDX)

    def get_loss(self, data, reduction=True):
        src, src_mask, trg, _ = data
        out = self.forward(src, src_mask, trg)
        loss = self.loss(out, trg[1:], reduction=reduction)
        return loss


HMMState = namedtuple("HMMState", "init trans emiss")


class HMM(object):
    def __init__(self, nb_states, nb_tokens, initial, transition, emission):
        assert isinstance(initial, torch.Tensor)
        assert isinstance(transition, torch.Tensor)
        assert isinstance(emission, torch.Tensor)
        assert initial.shape[-1] == nb_states
        assert transition.shape[-2:] == (nb_states, nb_states)
        assert emission.shape[-2:] == (nb_states, nb_tokens)
        self.ns = nb_states
        self.V = nb_tokens
        self.initial = initial
        self.transition = transition
        self.emission = emission

    def emiss(self, T, idx, ignore_index=None):
        assert len(idx.shape) == 1
        bs = idx.shape[0]
        idx = idx.view(-1, 1).expand(bs, self.ns).unsqueeze(-1)
        emiss = torch.gather(self.emission[T], -1, idx).view(bs, 1, self.ns)
        if ignore_index is None:
            return emiss
        else:
            idx = idx.view(bs, 1, self.ns)
            mask = (idx != ignore_index).float()
            return emiss * mask

    def p_x(self, seq, ignore_index=None):
        T, bs = seq.shape
        assert self.initial.shape == (bs, 1, self.ns)
        assert self.transition.shape == (T - 1, bs, self.ns, self.ns)
        assert self.emission.shape == (T, bs, self.ns, self.V)
        # fwd = pi * b[:, O[0]]
        # fwd = self.initial * self.emiss(0, seq[0])
        fwd = self.initial + self.emiss(0, seq[0], ignore_index=ignore_index)
        # induction:
        for t in range(T - 1):
            # fwd[t + 1] = np.dot(fwd[t], a) * b[:, O[t + 1]]
            # fwd = torch.bmm(fwd, self.transition[t]) * self.emiss(
            #     t + 1, seq[t + 1])
            fwd = fwd + self.transition[t].transpose(1, 2)
            fwd = fwd.logsumexp(dim=-1, keepdim=True).transpose(1, 2)
            fwd = fwd + self.emiss(t + 1, seq[t + 1], ignore_index=ignore_index)
        return fwd


class HMMTransducer(Transducer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.attn

    def loss(self, predict, target, reduction=True):
        assert isinstance(predict, HMMState)
        seq_len = target.shape[0]
        hmm = HMM(
            predict.init.shape[-1],
            self.trg_vocab_size,
            predict.init,
            predict.trans,
            predict.emiss,
        )
        loss = hmm.p_x(target, ignore_index=PAD_IDX)
        if not reduction:
            return -torch.logsumexp(loss, dim=-1) / seq_len
        return -torch.logsumexp(loss, dim=-1).mean() / seq_len

    def decode(self, enc_hs, enc_mask, trg_batch):
        trg_seq_len = trg_batch.size(0)
        trg_bat_siz = trg_batch.size(1)
        trg_embed = self.dropout(self.trg_embed(trg_batch))
        hidden = self.dec_rnn.get_init_hx(trg_bat_siz)

        initial, transition, emission = None, list(), list()
        for idx in range(trg_seq_len - 1):
            input_ = trg_embed[idx, :]
            trans, emiss, hidden = self.decode_step(enc_hs, enc_mask, input_, hidden)
            if idx == 0:
                initial = trans[:, 0].unsqueeze(1)
                emission += [emiss]
            else:
                transition += [trans]
                emission += [emiss]
        transition = torch.stack(transition)
        emission = torch.stack(emission)
        return HMMState(initial, transition, emission)

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        src_seq_len, bat_siz = enc_mask.shape
        h_t, hidden = self.dec_rnn(input_, hidden)

        # Concatenate the ht and hs
        # ctx_*: batch x seq_len x (trg_hid_siz+src_hid_size*2)
        ctx_curr = torch.cat(
            (h_t.unsqueeze(1).expand(-1, src_seq_len, -1), enc_hs[0].transpose(0, 1)),
            dim=2,
        )

        hs_ = enc_hs[1].transpose(0, 1)
        h_t = h_t.unsqueeze(2)
        score = torch.bmm(hs_, h_t).squeeze(2)
        trans = F.softmax(score, dim=-1) * enc_mask.transpose(0, 1) + EPSILON
        trans = trans / trans.sum(-1, keepdim=True)
        trans = trans.unsqueeze(1).log()
        trans = trans.expand(bat_siz, src_seq_len, src_seq_len)

        ctx = torch.tanh(self.linear_out(ctx_curr))
        # emiss: batch x seq_len x nb_vocab
        emiss = F.log_softmax(self.final_out(ctx), dim=-1)

        return trans, emiss, hidden


class FullHMMTransducer(HMMTransducer):
    def __init__(self, wid_siz, **kwargs):
        super().__init__(**kwargs)
        assert wid_siz % 2 == 1
        self.wid_siz = wid_siz
        self.trans = nn.Linear(self.trg_hid_size * 2, self.wid_siz)

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        src_seq_len, bat_siz = enc_mask.shape
        h_t, hidden = self.dec_rnn(input_, hidden)

        # Concatenate the ht and hs
        # ctx_trans: batch x seq_len x (trg_hid_siz*2)
        ctx_trans = torch.cat(
            (h_t.unsqueeze(1).expand(-1, src_seq_len, -1), enc_hs[1].transpose(0, 1)),
            dim=2,
        )
        trans = F.softmax(self.trans(ctx_trans), dim=-1)
        trans_list = trans.split(1, dim=1)
        ws = (self.wid_siz - 1) // 2
        trans_shift = [
            F.pad(t, (-ws + i, src_seq_len - (ws + 1) - i))
            for i, t in enumerate(trans_list)
        ]
        trans = torch.cat(trans_shift, dim=1)
        trans = trans * enc_mask.transpose(0, 1).unsqueeze(1) + EPSILON
        trans = trans / trans.sum(-1, keepdim=True)
        trans = trans.log()

        # Concatenate the ht and hs
        # ctx_emiss: batch x seq_len x (trg_hid_siz+src_hid_size*2)
        ctx_emiss = torch.cat(
            (h_t.unsqueeze(1).expand(-1, src_seq_len, -1), enc_hs[0].transpose(0, 1)),
            dim=2,
        )
        ctx = torch.tanh(self.linear_out(ctx_emiss))
        # emiss: batch x seq_len x nb_vocab
        emiss = F.log_softmax(self.final_out(ctx), dim=-1)

        return trans, emiss, hidden


class MonoHMMTransducer(HMMTransducer):
    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        trans, emiss, hidden = super().decode_step(enc_hs, enc_mask, input_, hidden)
        trans_mask = torch.ones_like(trans[0]).triu().unsqueeze(0)
        trans_mask = (trans_mask - 1) * -np.log(EPSILON)
        trans = trans + trans_mask
        trans = trans - trans.logsumexp(-1, keepdim=True)
        return trans, emiss, hidden


class HardMonoTransducer(Transducer):
    def __init__(self, *, nb_attr, **kwargs):
        super().__init__(**kwargs)
        self.nb_attr = nb_attr + 1 if nb_attr > 0 else 0
        # StackedLSTM(embed_dim, trg_hid_size, trg_nb_layers, dropout_p)
        hs = self.cal_hs(
            layer=self.trg_nb_layers,
            ed=self.embed_dim,
            od=self.out_dim,
            vs=self.trg_vocab_size,
            hs=self.src_hid_size,
            ht=self.trg_hid_size,
        )
        if self.nb_attr > 0:
            self.merge_attr = nn.Linear(self.embed_dim * self.nb_attr, self.embed_dim)
            self.dec_rnn = StackedLSTM(
                self.embed_dim * 2 + self.src_hid_size * 2,
                hs,
                self.trg_nb_layers,
                self.dropout_p,
            )
        else:
            self.dec_rnn = StackedLSTM(
                self.embed_dim + self.src_hid_size * 2,
                hs,
                self.trg_nb_layers,
                self.dropout_p,
            )
        # nn.Linear(self.out_dim, trg_vocab_size)
        self.final_out = nn.Linear(hs, self.trg_vocab_size)
        del self.scale_enc_hs  # nn.Linear(src_hid_size * 2, trg_hid_size)
        del self.attn
        del self.linear_out  # nn.Linear(self.out_dim, self.out_dim)

    def cal_hs(self, *, layer, ed, od, vs, hs, ht):
        b = ed + 2 * hs + 2 + vs / 4
        if self.nb_attr > 0:
            b += ed
        c = (
            ed * ed * self.nb_attr
            + ed
            - od * (od + vs + 1)
            - ht * (2 * hs + 4 * ht + 4 * ed + 1 + 4 * 2)
        )
        c /= 4
        if layer > 1:
            c -= (layer - 1) * (2 * ht**2 + 2 * ht)
            b += (layer - 1) * 2
            b /= layer * 2 - 1
            c /= layer * 2 - 1
        return round((math.sqrt(b * b - 4 * c) - b) / 2)

    def encode(self, src_batch):
        """
        encoder
        """
        if self.nb_attr > 0:
            assert isinstance(src_batch, tuple) and len(src_batch) == 2
            src, attr = src_batch
            bs = src.shape[1]
            enc_hs, _ = self.enc_rnn(self.dropout(self.src_embed(src)))
            enc_attr = F.relu(self.merge_attr(self.src_embed(attr).view(bs, -1)))
            return enc_hs, enc_attr
        else:
            enc_hs, _ = self.enc_rnn(self.dropout(self.src_embed(src_batch)))
            return enc_hs, None

    def decode_step(self, enc_hs, enc_mask, input_, hidden, attn_pos):
        """
        decode step
        """
        source, attr = enc_hs
        bs = source.shape[1]
        if isinstance(attn_pos, int):
            assert bs == 1
            ctx = source[attn_pos]
        else:
            ctx = fancy_gather(source, attn_pos).squeeze(0)
        if attr is None:
            input_ = torch.cat((input_, ctx), dim=1)
        else:
            input_ = torch.cat((input_, attr, ctx), dim=1)
        h_t, hidden = self.dec_rnn(input_, hidden)
        word_logprob = F.log_softmax(self.final_out(h_t), dim=-1)
        return word_logprob, hidden, None

    def decode(self, enc_hs, enc_mask, trg_batch):
        """
        enc_hs: tuple(enc_hs, enc_attr)
        """
        trg_seq_len = trg_batch.size(0)
        trg_bat_siz = trg_batch.size(1)
        attn_pos = torch.zeros((1, trg_bat_siz), dtype=torch.long, device=DEVICE)
        trg_embed = self.dropout(self.trg_embed(trg_batch))
        output = []
        hidden = self.dec_rnn.get_init_hx(trg_bat_siz)
        for idx in range(trg_seq_len - 1):
            # for j in range(trg_bat_siz):
            #     if trg_batch[idx, j] == STEP_IDX:
            #         attn_pos[0, j] += 1
            attn_pos = attn_pos + (trg_batch[idx] == STEP_IDX)
            input_ = trg_embed[idx, :]
            word_logprob, hidden, _ = self.decode_step(
                enc_hs, enc_mask, input_, hidden, attn_pos
            )
            output += [word_logprob]
        return torch.stack(output)


class InputFeedTransducer(Transducer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("previous size\n{}\n{}".format(self.linear_out, self.final_out))
        self.scale_out = self.calculate_scale_out(
            self.out_dim, self.trg_vocab_size, self.embed_dim
        )
        self.linear_out = nn.Linear(self.out_dim, self.scale_out)
        self.final_out = nn.Linear(self.scale_out, self.trg_vocab_size)
        self.merge_input = nn.Linear(self.embed_dim + self.scale_out, self.embed_dim)
        print(
            "new size\n{}\n{}\n{}".format(
                self.linear_out, self.final_out, self.merge_input
            )
        )

    def calculate_scale_out(self, od, vt, e):
        num = od * od + od + od * vt - e * e - e
        den = e + 1 + od + vt
        return round(num / den)

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        """
        decode step
        """
        bs = input_.shape[0]
        if isinstance(hidden[0], tuple):
            prev_hidden, prev_context = hidden
        else:
            prev_hidden = hidden
            prev_context = torch.zeros((bs, self.scale_out), device=DEVICE)
        input_ = self.merge_input(torch.cat((input_, prev_context), dim=1))
        h_t, hidden = self.dec_rnn(input_, prev_hidden)
        ctx, attn = self.attn(h_t, enc_hs, enc_mask)
        # Concatenate the ht and ctx
        # weight_hs: batch x (hs_dim + ht_dim)
        ctx = torch.cat((ctx, h_t), dim=1)
        # ctx: batch x out_dim
        ctx = self.linear_out(ctx)
        ctx = torch.tanh(ctx)
        hidden = (hidden, ctx)
        word_logprob = F.log_softmax(self.final_out(ctx), dim=-1)
        return word_logprob, hidden, attn


class LargeInputFeedTransducer(InputFeedTransducer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # print('previous size\n{}\n{}\n{}'.format(self.linear_out, self.final_out, self.merge_input))
        self.scale_out = self.out_dim
        self.linear_out = nn.Linear(self.out_dim, self.out_dim)
        self.final_out = nn.Linear(self.out_dim, self.trg_vocab_size)
        self.merge_input = Identity()
        self.dec_rnn = StackedLSTM(
            self.embed_dim + self.out_dim,
            self.trg_hid_size,
            self.trg_nb_layers,
            self.dropout_p,
        )
        print(
            "new size\n{}\n{}\n{}".format(
                self.linear_out, self.final_out, self.merge_input
            )
        )


class HardAttnTransducer(Transducer):
    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        """
        enc_hs: tuple(enc_hs, scale_enc_hs)
        """
        src_seq_len = enc_hs[0].size(0)
        h_t, hidden = self.dec_rnn(input_, hidden)

        # ht: batch x trg_hid_dim
        # enc_hs: seq_len x batch x src_hid_dim*2
        # attns: batch x 1 x seq_len
        _, attns = self.attn(h_t, enc_hs, enc_mask, weighted_ctx=False)

        # Concatenate the ht and hs
        # ctx: batch x seq_len x (trg_hid_siz+src_hid_size*2)
        ctx = torch.cat(
            (h_t.unsqueeze(1).expand(-1, src_seq_len, -1), enc_hs[0].transpose(0, 1)),
            dim=2,
        )
        # ctx: batch x seq_len x out_dim
        ctx = self.linear_out(ctx)
        ctx = torch.tanh(ctx)

        # word_prob: batch x seq_len x nb_vocab
        word_prob = F.softmax(self.final_out(ctx), dim=-1)
        # word_prob: batch x nb_vocab
        word_prob = torch.bmm(attns, word_prob).squeeze(1)
        return torch.log(word_prob), hidden, attns


class TagTransducer(Transducer):
    def __init__(self, *, nb_attr, **kwargs):
        super().__init__(**kwargs)
        self.nb_attr = nb_attr + 1 if nb_attr > 0 else 0
        if self.nb_attr > 0:
            attr_dim = self.embed_dim // 5
            self.src_embed = nn.Embedding(
                self.src_vocab_size - nb_attr, self.embed_dim, padding_idx=PAD_IDX
            )
            # padding_idx is a part of self.nb_attr, so need to +1
            self.attr_embed = nn.Embedding(
                self.nb_attr + 1, attr_dim, padding_idx=PAD_IDX
            )
            self.merge_attr = nn.Linear(attr_dim * self.nb_attr, attr_dim)
            self.dec_rnn = StackedLSTM(
                self.embed_dim + attr_dim,
                self.trg_hid_size,
                self.trg_nb_layers,
                self.dropout_p,
            )

    def encode(self, src_batch):
        """
        encoder
        """
        if self.nb_attr > 0:
            assert isinstance(src_batch, tuple) and len(src_batch) == 2
            src, attr = src_batch
            bs = src.shape[1]
            new_idx = torch.arange(1, self.nb_attr + 1).expand(bs, -1)
            attr = ((attr > 1).float() * new_idx.to(attr.device).float()).long()
            enc_attr = F.relu(self.merge_attr(self.attr_embed(attr).view(bs, -1)))
        else:
            src = src_batch
            enc_attr = None
        enc_hs = super().encode(src)
        return enc_hs, enc_attr

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        """
        decode step
        """
        enc_hs_, attr = enc_hs
        if attr is not None:
            input_ = torch.cat((input_, attr), dim=1)
        return super().decode_step(enc_hs_, enc_mask, input_, hidden)


class TagHMMTransducer(TagTransducer, HMMTransducer):
    pass


class TagFullHMMTransducer(TagTransducer, FullHMMTransducer):
    pass


class MonoTagHMMTransducer(TagTransducer, MonoHMMTransducer):
    pass


class MonoTagFullHMMTransducer(TagTransducer, MonoHMMTransducer, FullHMMTransducer):
    pass


class TagHardAttnTransducer(TagTransducer, HardAttnTransducer):
    pass


def fancy_gather(value, index):
    assert value.size(1) == index.size(1)
    split = zip(value.split(1, dim=1), index.split(1, dim=1))
    return torch.cat([v[i.view(-1)] for v, i in split], dim=1)


class Categorical(Distribution):
    def __init__(self, probs):
        assert probs.dim() == 2
        self.nb_prob, self.nb_choice = probs.size()
        self.probs = probs
        self.probs_t = probs.t()

    def sample_n(self, n):
        return torch.multinomial(self.probs, n, True).t()

    def log_prob(self, value):
        return (fancy_gather(self.probs_t, value) + EPSILON).log()


class ApproxiHardTransducer(Transducer):
    def __init__(self, *, nb_sample, **kwargs):
        super().__init__(**kwargs)
        self.nb_sample = nb_sample
        self.log_probs = []
        self.aver_reward = 0
        self.disc = 0.9
        self.gamma = 1

    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        h_t, hidden = self.dec_rnn(input_, hidden)

        # ht: batch x trg_hid_dim
        # enc_hs: seq_len x batch x src_hid_dim*2
        # attns: batch x 1 x seq_len
        _, attns = self.attn(h_t, enc_hs, enc_mask, weighted_ctx=False)
        attns = attns.squeeze(1)
        sampler = Categorical(attns)
        index = sampler.sample_n(self.nb_sample)
        self.log_probs.append(sampler.log_prob(index))

        ctx = fancy_gather(enc_hs[0], index)
        ctx = torch.cat([h_t.unsqueeze(0).expand(self.nb_sample, -1, -1), ctx], dim=-1)
        ctx = torch.tanh(self.linear_out(ctx))
        # word_prob: nb_sample x batch x nb_vocab
        word_prob = F.softmax(self.final_out(ctx), dim=-1)
        word_prob = word_prob.transpose(0, 1)
        sel_attns = fancy_gather(attns.t(), index)
        sel_attns = sel_attns / sel_attns.sum(0, keepdim=True)
        sel_attns = sel_attns.t().unsqueeze(1)
        # mix_prob: batch x nb_vocab
        word_prob = torch.bmm(sel_attns, word_prob).squeeze(1)
        return torch.log(word_prob), hidden, attns

    def encode(self, src_batch):
        self.log_probs = []
        return super().encode(src_batch)

    def loss(self, predict, target, reduction=True):
        """
        compute loss
        """
        nll_loss = F.nll_loss(
            predict.view(-1, self.trg_vocab_size),
            target.view(-1),
            ignore_index=PAD_IDX,
            reduce=False,
        )

        if not reduction:
            nll_loss = nll_loss.view(target.shape)
            nll_loss = nll_loss.sum(dim=0) / (target != PAD_IDX).sum(dim=0)
            return nll_loss

        policy_loss = []
        for log_prob, reward in zip(self.log_probs, nll_loss):
            policy_loss.append(-log_prob * (reward - self.aver_reward))
        policy_loss = torch.cat(policy_loss).mean()
        nll_loss = nll_loss.mean()
        self.aver_reward = (
            self.disc * self.aver_reward + (1 - self.disc) * nll_loss.item()
        )
        return policy_loss * self.gamma + nll_loss


class ApproxiHardInputFeedTransducer(ApproxiHardTransducer, InputFeedTransducer):
    def decode_step(self, enc_hs, enc_mask, input_, hidden):
        bs = input_.shape[0]
        if isinstance(hidden[0], tuple):
            prev_hidden, prev_context = hidden
        else:
            prev_hidden = hidden
            prev_context = torch.zeros((bs, self.scale_out), device=DEVICE)
        input_ = self.merge_input(torch.cat((input_, prev_context), dim=1))
        h_t, hidden = self.dec_rnn(input_, prev_hidden)

        # ht: batch x trg_hid_dim
        # enc_hs: seq_len x batch x src_hid_dim*2
        # attns: batch x 1 x seq_len
        _, attns = self.attn(h_t, enc_hs, enc_mask, weighted_ctx=False)
        attns = attns.squeeze(1)
        sampler = Categorical(attns)
        index = sampler.sample_n(self.nb_sample)
        self.log_probs.append(sampler.log_prob(index))

        ctx = fancy_gather(enc_hs[0], index).transpose(0, 1)
        sel_attns = fancy_gather(attns.t(), index)
        sel_attns = sel_attns / sel_attns.sum(0, keepdim=True)
        sel_attns = sel_attns.t().unsqueeze(1)
        ctx = torch.bmm(sel_attns, ctx).squeeze(1)

        ctx = torch.cat((ctx, h_t), dim=1)
        ctx = torch.tanh(self.linear_out(ctx))
        hidden = (hidden, ctx)
        word_logprob = F.log_softmax(self.final_out(ctx), dim=-1)
        return word_logprob, hidden, attns.unsqueeze(1)


def dummy_mask(seq):
    """
    create dummy mask (all 1)
    """
    if isinstance(seq, tuple):
        seq = seq[0]
    return torch.ones_like(seq, dtype=torch.float)
