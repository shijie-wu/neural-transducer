# Hard Non-Monotonic Attention for Character-Level Transduction

Shijie Wu, Pamela Shapiro, and Ryan Cotterell. [*Hard Non-Monotonic Attention for Character-Level Transduction*](https://arxiv.org/abs/1808.10024). EMNLP. 2018.

## Experiments

We use morphological inflection as an example since the data is hosted on GitHub. Feel free to contact shijie.wu_at_jhu.edu for g2p and transliteration.

```bash
# We use latin as an example
lang=latin
# Model size: large (or small)
model=large

# download this repo
git clone https://github.com/shijie-wu/neural-transducer.git
cd neural-transducer

# download data
mkdir data
cd data
git clone https://github.com/sigmorphon/conll2017.git
cd ..

# Run soft attention with input feeding (1)
sh example/hard-attention/sigmorphon17task1/run-sigmorphon17task1-$model.sh softinputfeed $lang

# Run hard attention with REINFORCE approximation with input-feeding (2)
sh example/hard-attention/sigmorphon17task1/run-sigmorphon17task1-$model.sh approxihardinputfeed $lang

# Run soft attention without input feeding (3)
sh example/hard-attention/sigmorphon17task1/run-sigmorphon17task1-$model.sh soft $lang

# Run hard attention without input feeding (4) (OUR)
sh example/hard-attention/sigmorphon17task1/run-sigmorphon17task1-$model.sh hard $lang

# Run monotonic hard attention by Aharoni and Goldberg (2017) (M)
sh example/hard-attention/sigmorphon17task1/run-sigmorphon17task1-$model.sh hardmono $lang

# Run variant of soft attention with input feeding where the number of parameters is not controlled  (U)
sh example/hard-attention/sigmorphon17task1/run-sigmorphon17task1-$model.sh largesoftinputfeed $lang

# Run variant of hard attention without input feeding using REINFORCE instead of exact marginalization (R)
sh example/hard-attention/sigmorphon17task1/run-sigmorphon17task1-$model.sh approxihard $lang
```
