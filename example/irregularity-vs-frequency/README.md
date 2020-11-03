# Morphological Irregularity Correlates with Frequency

Shijie Wu, Ryan Cotterell, and Timothy J O'Donnell. [*Morphological Irregularity Correlates with Frequency*](https://arxiv.org/abs/1906.11483). ACL. 2019.

## Run Experiments on UniMorph

### Train Models


```bash
lang=eng # English as an example
nfold=10 # number of folds

# download this repo
git clone https://github.com/shijie-wu/neural-transducer.git
cd neural-transducer

# download unimorph
mkdir -p data/unimorph
git clone https://github.com/unimorph/$lang.git data/unimorph/$lang

# prepare data
python example/irregularity-vs-frequency/preprocess-unimorph.py \
    --infile data/unimorph/$lang/$lang \
    --outdir data/unimorph/$lang/split/$lang \
    --nfold 10 --nchar 1 --prefix --suffix

# train models
for fold in $(seq 1 $nfold); do
sh example/irregularity-vs-frequency/run-unimorph.sh $lang $fold
done
```

The output in `model/unimorph/large/monotag-hmm/{lang}-{fold}.decode.test.tsv` (the `loss` column) should contains `-log( p(inflected form|lemma, tags) ) / (len(inflected form) + 1)`, including a special end-of-sequence token `</s>`.

### Count WikiPedia

Note it might take a while.

```bash
# download wikipedia, take English as an example
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# install package
pip install gensim langcodes nltk smart_open

# cleanup wikipedia
python -m gensim.scripts.segment_wiki -i -f enwiki-latest-pages-articles.xml.bz2 -o enwiki-latest.json.gz

# count wikipedia
python example/irregularity-vs-frequency/count-wiki.py \
    --wiki enwiki-latest.json.gz \
    --testfiles data/unimorph/eng/split/eng.* \
    --lang en \
    --outfile data/unimorph/eng/split/eng.wiki.cnt
```

## Run Experiments on English Past Tense

```bash
# download Albright and Hayes (2003)
wget https://www.dropbox.com/s/26mqdxx94r1l72d/english_merged.txt -O data/unimorph/eng.albright/eng.albright

# download O’Donnell (2015)
wget https://www.dropbox.com/s/wccovnvhaz0tpq3/pasttense-adult-PTB.verbs.csv -O data/unimorph/eng.odonnell/eng.odonnell

# prepare data
for mode in albright odonnell; do
python example/irregularity-vs-frequency/preprocess-eng-past.py \
    --unimorph data/unimorph/eng/eng \
    --past data/unimorph/eng.$mode/eng.$mode \
    --outdir data/unimorph/eng.$mode/eng.$mode \
    --nchar 1 --prefix --suffix --mode $mode
done

# train models
for mode in albright odonnell; do
sh example/irregularity-vs-frequency/run-eng-past.sh $mode
done
```
