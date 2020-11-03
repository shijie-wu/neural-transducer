# Simple class for learning an alignment of strings, MED-style.
# Weights are learned by a Chinese Restaurant Process sampler
# that weights single alignments x:y in proportion to how many times
# such an alignment has been seen elsewhere out of all possible alignments.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

# Usage:
# Align(wordpairs) <= wordpairs is an iterable of 2-tuples
# The resulting Align.alignedpairs is a list of aligned 2-tuples

# Relies on C-code in libalign.so built from align.c through ctypes.
# Author: Mans Hulden
# MH20151102

import itertools
from ctypes import POINTER, c_int, c_void_p, cdll

libalign = cdll.LoadLibrary("src/libalign.so")

libalign_add_int_pair = libalign.add_int_pair
libalign_clear_counts = libalign.clear_counts
libalign_initial_align = libalign.initial_align
libalign_crp_train = libalign.crp_train
libalign_crp_align = libalign.crp_align
libalign_med_align = libalign.med_align

libalign_getpairs_init = libalign.getpairs_init
libalign_getpairs_init.restype = c_void_p
libalign_getpairs_in = libalign.getpairs_in
libalign_getpairs_in.restype = POINTER(c_int)
libalign_getpairs_out = libalign.getpairs_out
libalign_getpairs_out.restype = POINTER(c_int)
libalign_getpairs_advance = libalign.getpairs_advance
libalign_getpairs_advance.restype = c_void_p
libalign_align_init = libalign.align_init
libalign_align_init.restype = None


class Aligner:
    def __init__(
        self, wordpairs, align_symbol="~", iterations=10, burnin=5, lag=1, mode="crp"
    ):
        s = set()
        for x, y in wordpairs:
            s.update(x)
            s.update(y)
        self.symboltoint = dict(list(zip(s, list(range(1, len(s) + 1)))))
        self.inttosymbol = {v: k for k, v in list(self.symboltoint.items())}
        assert len(self.symboltoint) < 4096
        self.inttosymbol[0] = align_symbol
        # Map stringpairs to -1 terminated integer sequences
        intpairs = []
        for i, o in wordpairs:
            intin = [self.symboltoint[x] for x in i] + [-1]
            intout = [self.symboltoint[x] for x in o] + [-1]
            intpairs.append((intin, intout))

        libalign_align_init()
        for i, o in intpairs:
            icint = (c_int * len(i))(*i)
            ocint = (c_int * len(o))(*o)
            libalign_add_int_pair(icint, ocint)

        # Run CRP align
        if mode == "crp":
            libalign_clear_counts()
            libalign_initial_align()
            libalign_crp_train(c_int(iterations), c_int(burnin), c_int(lag))
            libalign_crp_align()
        else:
            libalign_clear_counts()
            libalign_initial_align()
            libalign_med_align()

        # Reconvert to output
        self.alignedpairs = []
        stringpairptr = libalign_getpairs_init()
        while stringpairptr is not None:
            inints = libalign_getpairs_in(c_void_p(stringpairptr))
            outints = libalign_getpairs_out(c_void_p(stringpairptr))
            instr = []
            outstr = []
            for j in itertools.count():
                if inints[j] == -1:
                    break
                instr.append(self.inttosymbol[inints[j]])
            for j in itertools.count():
                if outints[j] == -1:
                    break
                outstr.append(self.inttosymbol[outints[j]])
            self.alignedpairs.append((instr, outstr))
            stringpairptr = libalign_getpairs_advance(c_void_p(stringpairptr))
