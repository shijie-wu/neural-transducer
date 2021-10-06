# SIGMORPHON 2021 Shared Task 0 Baseline

- Tiago Pimentel, Maria Ryskina, Sabrina J. Mielke, Shijie Wu, Eleanor Chodroff, Brian Leonard, Garrett Nicolai, Yustinus Ghanggo Ate, Salam Khalifa, Nizar Habash, Charbel El-Khaissi, Omer Goldman, Michael Gasser, William Lane, Matt Coler, Arturo Oncevay, Jaime Rafael Montoya Samame, Gema Celeste Silva Villegas, Adam Ek, Jean-Philippe Bernardy, Andrey Shcherbakov, Aziyana Bayyr-ool, Karina Sheifer, Sofya Ganieva, Matvey Plugaryov, Elena Klyachko, Ali Salehi, Andrew Krizhanovsky, Natalia Krizhanovsky, Clara Vania, Sardana Ivanova, Aelita Salchak, Christopher Straughn, Zoey Liu, Jonathan North Washington, Duygu Ataman, Witold Kieraś, Marcin Woliński, Totok Suhardijanto, Niklas Stoehr, Zahroh Nuriah, Shyam Ratan, Francis M. Tyers, Edoardo M. Ponti, Grant Aiton, Richard J. Hatcher, Emily Prud'hommeaux, Ritesh Kumar, Mans Hulden, Botond Barta, Dorina Lakatos, Gábor Szolnok, Judit Ács, Mohit Raj, David Yarowsky, Ryan Cotterell, Ben Ambridge, and Ekaterina Vylomova. [*SIGMORPHON 2021 Shared Task on Morphological Reinflection: Generalization Across Languages*](https://aclanthology.org/2021.sigmorphon-1.25/). SIGMORPHON. 2021.


### Training from Scratch

First download and (optionally) augment [(Anastasopoulos and Neubig, 2019)](https://arxiv.org/abs/1908.05838) the data for training set smaller than 10k.

```bash
git clone https://github.com/sigmorphon/2021Task0.git

for lng in ail ame bra ckt evn gup itl kod lud mag see syc; do
  python example/sigmorphon2020-shared-tasks/augment.py \
    2021Task0/part1/development_languages $lng --examples 10000
  python example/sigmorphon2021-shared-tasks/build-dataset.py \
    2021Task0/part1/development_languages $lng
done
for lng in sjo vro; do
  python example/sigmorphon2020-shared-tasks/augment.py \
    2021Task0/part1/surprise_languages $lng --examples 10000
  python example/sigmorphon2021-shared-tasks/build-dataset.py \
    2021Task0/part1/surprise_languages $lng
done
```

Run input-invariant transformer [(Wu and Cotterell, 2019)](https://arxiv.org/abs/2005.10213) or the vanilla transformer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762).
```bash
model=tagtransformer # input-invariant transformer, or `model=transformer` the vanilla transformer
# No data augmentation
for lang in afb ail ame amh ara arz aym bra bul ces ckb ckt cni deu evn gup heb ind itl kmr kod krl lud mag nld olo pol por rus sah see sjo spa syc tur tyv vep vro; do
bash example/sigmorphon2021-shared-tasks/part1-trm.sh $lang $model
done

# Data augmentation (training size < 10k)
for lang in ail ame bra ckt evn gup itl kod lud mag see syc sjo vro; do
bash example/sigmorphon2021-shared-tasks/part1-trm-hall.sh $lang $model
done
```
