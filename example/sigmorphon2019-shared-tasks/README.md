# SIGMORPHON 2019 Shared Tasks Baseline

Arya D McCarthy, Ekaterina Vylomova, Shijie Wu, Chaitanya Malaviya, Lawrence Wolf-Sonkin, Garrett Nicolai, Miikka Silfverberg, Sebastian J Mielke, Jeffrey Heinz, Ryan Cotterell, and Mans Hulden. [*The SIGMORPHON 2019 Shared Task: Morphological Analysis in Context and Cross-Lingual Transfer for Inflection*](https://www.aclweb.org/anthology/W19-4226/). SIGMORPHON. 2019.


## Task 1: crosslingual-inflection-baseline

### Baseline Performance

Available for all baselines [here](https://docs.google.com/spreadsheets/d/1vvSuy2LBarS20zK8lg_YCTauntDsmoxfqSaSrAQsJrM/edit?usp=sharing).

### Training from Scratch

```bash
dir=example/sigmorphon2019-shared-tasks
# 0-soft
sh $dir/run-task1-tag.sh soft adyghe--kabardian
# 0-hard
sh $dir/run-task1-tag.sh hard adyghe--kabardian
# 0-mono
sh $dir/run-task1-monotag.sh hmm adyghe--kabardian
# 1-mono
sh $dir/run-task1-monotag.sh hmmfull adyghe--kabardian
```

### Decoding with Pretrained Model

```bash
dir=example/sigmorphon2019-shared-tasks
python src/sigmorphon19-task1-decode.py \
    --in_file $dir/sample/task1/adyghe--kabardian/kabardian-dev \
    --out_file decode/task1/adyghe--kabardian-dev-out \
    --lang kabardian \
    --model $dir/sample/task1/model/adyghe--kabardian.1-mono.pth
```

## Task 2: contextual-analysis-baseline

### Baseline Performance

Available for all baselines [here](https://docs.google.com/spreadsheets/d/1R1dtj2YFhPaOv4-VE1TpcCJ5_WzKO6rZ8ObMxJsM020/edit?usp=sharing).

Task 2 decoded files: https://www.dropbox.com/s/2kqkhsu0kil6rzu/BASELINE-DEV-00-2.tar.gz

### Training from Scratch

We train the model with the [jackknifing training data](https://www.dropbox.com/s/swf9cq22uxgr5wv/task2_jackknife_training_data_public.tar.gz) and at dev time, we decode the lemma with [predicted tag](https://www.dropbox.com/s/qt6nqa3gn96rbl3/baseline_predictions_public.tar.gz).

```bash
dir=example/sigmorphon2019-shared-tasks
# 0-mono
sh $dir/run-task2.sh af_afribooms
```

### Decoding with Pretrained Model

```bash
dir=example/sigmorphon2019-shared-tasks
python src/sigmorphon19-task2-decode.py \
    --in_file $dir/sample/task2/af_afribooms-um-dev.conllu.baseline.pred \
    --out_file decode/task2/af_afribooms-um-dev.conllu.output \
    --model $dir/sample/task2/model/af_afribooms.pth
```

### Pretrained Models

Link: https://www.dropbox.com/sh/p4vu5imyn69wyyp/AAA-3bQeGJmnCex78xx7T0ZPa

Size of models:
```
3.4G	sigmorphon2019/public/task2/lemmatizer-model
```
