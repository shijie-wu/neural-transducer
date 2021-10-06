# Applying the Transformer to Character-level Transduction

Shijie Wu, Ryan Cotterell, and Mans Hulden. [*Applying the Transformer to Character-level Transduction*](https://arxiv.org/abs/2005.10213). EACL. 2021.

## Experiments

We use morphological inflection as an example since the data is hosted on GitHub. Feel free to contact shijie.wu_at_jhu.edu for g2p and transliteration. The historical text normalization dataset can be downloaded [here](https://github.com/coastalcph/histnorm)

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

# Run feature-invariant transformer (`arch=tagtransformer`). For regular transformer, `arch=transformer`.
sh example/transformer/trm-sig17.sh $lang
```
