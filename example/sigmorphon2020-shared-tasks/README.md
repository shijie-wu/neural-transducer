# SIGMORPHON 2020 Shared Task 0 Baseline

- Ekaterina Vylomova, Jennifer White, Elizabeth Salesky, Sabrina J. Mielke, Shijie Wu, Edoardo Ponti, Rowan Hall Maudslay, Ran Zmigrod, Josef Valvoda, Svetlana Toldova, Francis Tyers, Elena Klyachko, Ilya Yegorov, Natalia Krizhanovsky, Paula Czarnowska, Irene Nikkarinen, Andrew Krizhanovsky, Tiago Pimentel, Lucas Torroba Hennigen, Christo Kirov, Garrett Nicolai, Adina Williams, Antonios Anastasopoulos, Hilaria Cruz, Eleanor Chodroff, Ryan Cotterell, Miikka Silfverberg, and Mans Hulden. [*SIGMORPHON 2020 Shared Task 0: Typologically Diverse Morphological Inflection*](https://www.aclweb.org/anthology/2020.sigmorphon-1.1/). SIGMORPHON. 2020. ([Experiments Detail](example/sigmorphon2020-shared-tasks))


### Training from Scratch

First download and augment [(Anastasopoulos and Neubig, 2019)](https://arxiv.org/abs/1908.05838) the data

```bash
git clone https://github.com/sigmorphon2020/task0-data.git
mkdir task0-data/original
mv task0-data/DEVELOPMENT-LANGUAGES/*/* task0-data/original
mv task0-data/SURPRISE-LANGUAGES/*/* task0-data/original
mv task0-data/GOLD-TEST/* task0-data/original -f

bash example/sigmorphon2020-shared-tasks/augment.sh
python example/sigmorphon2020-shared-tasks/task0-build-dataset.py all
```

Run hard monotonic attention [(Wu and Cotterell, 2019)](https://arxiv.org/abs/1905.06319) and the transformer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), both one model per language and one model per language family.
```bash
bash example/sigmorphon2020-shared-tasks/task0-launch.sh
```
