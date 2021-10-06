# SIGMORPHON 2020 Shared Task 0 Baseline

- Ekaterina Vylomova, Jennifer White, Elizabeth Salesky, Sabrina J. Mielke, Shijie Wu, Edoardo Ponti, Rowan Hall Maudslay, Ran Zmigrod, Josef Valvoda, Svetlana Toldova, Francis Tyers, Elena Klyachko, Ilya Yegorov, Natalia Krizhanovsky, Paula Czarnowska, Irene Nikkarinen, Andrew Krizhanovsky, Tiago Pimentel, Lucas Torroba Hennigen, Christo Kirov, Garrett Nicolai, Adina Williams, Antonios Anastasopoulos, Hilaria Cruz, Eleanor Chodroff, Ryan Cotterell, Miikka Silfverberg, and Mans Hulden. [*SIGMORPHON 2020 Shared Task 0: Typologically Diverse Morphological Inflection*](https://www.aclweb.org/anthology/2020.sigmorphon-1.1/). SIGMORPHON. 2020.


### Training from Scratch

First download and (optionally) augment [(Anastasopoulos and Neubig, 2019)](https://arxiv.org/abs/1908.05838) the data

```bash
git clone https://github.com/sigmorphon2020/task0-data.git
mkdir task0-data/original
mv task0-data/DEVELOPMENT-LANGUAGES/*/* task0-data/original
mv task0-data/SURPRISE-LANGUAGES/*/* task0-data/original
mv task0-data/GOLD-TEST/* task0-data/original -f

langs=(aka ang ast aze azg bak ben bod cat ceb cly cpa cre crh ctp czn dak dan deu dje eng est evn fas fin frm frr fur gaa glg gmh gml gsw hil hin isl izh kan kaz kir kjh kon kpv krl lin liv lld lud lug mao mdf mhr mlg mlt mwf myv nld nno nob nya olo ood orm ote otm pei pus san sme sna sot swa swe syc tel tgk tgl tuk udm uig urd uzb vec vep vot vro xno xty zpv zul)

# (optionally) data augmentation
for lang in $langs; do
  python example/sigmorphon2020-shared-tasks/augment.py task0-data/original $lang --examples 10000
done

# no data augmentation
python example/sigmorphon2020-shared-tasks/task0-build-dataset.py regular
# data augmentation
python example/sigmorphon2020-shared-tasks/task0-build-dataset.py hall
# concat all language of the same family
python example/sigmorphon2020-shared-tasks/task0-build-dataset.py concat
# concat all language of the same family + data augmentation
python example/sigmorphon2020-shared-tasks/task0-build-dataset.py concat_hall
```

Run hard monotonic attention [(Wu and Cotterell, 2019)](https://arxiv.org/abs/1905.06319) and the transformer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), both one model per language and one model per language family.
```bash
model=mono # hard monotonic attention, or `model=trm` the vanilla transformer

# no data augmentation (one model per language)
for lang in $langs; do
  bash example/sigmorphon2020-shared-tasks/task0-$model.sh $lang
done

# data augmentation (one model per language)
for lang in $langs; do
  bash example/sigmorphon2020-shared-tasks/task0-hall-$model.sh $lang
done

# concat all language of the same family (one model per language family)
for family in afro-asiatic austronesian dravidian germanic indo-aryan iranian niger-congo oto-manguean romance turkic uralic; do
  bash example/sigmorphon2020-shared-tasks/task0-$model.sh $family
done

# concat all language of the same family + data augmentation (one model per language family)
for family in afro-asiatic austronesian dravidian germanic indo-aryan iranian niger-congo oto-manguean romance turkic uralic; do
  bash example/sigmorphon2020-shared-tasks/task0-hall-$model.sh $family
done
```
