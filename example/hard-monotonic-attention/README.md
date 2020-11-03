# Exact Hard Monotonic Attention for Character-Level Transduction

Shijie Wu, and Ryan Cotterell. [*Exact Hard Monotonic Attention for Character-Level Transduction*](https://arxiv.org/abs/1905.06319). ACL. 2019.

## Experiments

We use morphological inflection as an example since the data is hosted on GitHub. Feel free to contact shijie.wu_at_jhu.edu for g2p and transliteration.

```bash
# We use latin as an example
lang=latin

# download this repo
git clone https://github.com/shijie-wu/neural-transducer.git
cd neural-transducer

# download data
mkdir data
cd data
git clone https://github.com/sigmorphon/conll2017.git
cd ..

# Run soft attention (SOFT)
sh example/hard-monotonic-attention/sigmorphon17task1/run-sigmorphon17task1-large-tag.sh soft $lang

# Run 0th-order Hard Attentin (0-HARD)
sh example/hard-monotonic-attention/sigmorphon17task1/run-sigmorphon17task1-large-tag.sh hard $lang

# Run 0th-order Hard Monotonic Attentin (0-MONO) (OUR)
sh example/hard-monotonic-attention/sigmorphon17task1/run-sigmorphon17task1-large-monotag.sh hmm $lang

# Run 1st-order Hard Monotonic Attentin (1-MONO) (OUR)
sh example/hard-monotonic-attention/sigmorphon17task1/run-sigmorphon17task1-large-monotag.sh hmmfull $lang
```
