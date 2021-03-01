# SIGMORPHON 2021 Shared Task

## Task 0 Baseline

First download and augment [(Anastasopoulos and Neubig, 2019)](https://arxiv.org/abs/1908.05838) the data
```bash
git clone https://github.com/sigmorphon2021/task0-data.git
bash example/sigmorphon2021-shared-tasks/augment.sh
python example/sigmorphon2021-shared-tasks/task0-build-dataset.py all
```

Run the transducer [(Wu et al, 2021)](https://arxiv.org/abs/2005.10213), both one model per language and one model per language family.
```bash
bash example/sigmorphon2021-shared-tasks/task0-launch.sh
```


## Dependencies

- python 3
- pytorch==1.4
- numpy
- tqdm
- fire


## License

MIT

## References

- Shijie Wu, Ryan Cotterell, and Mans Hulden. [*Applying the Transformer to Character-level Transduction*](https://arxiv.org/abs/2005.10213). EACL. 2021.
- Antonios Anastasopoulos, and Graham Neubig. [*Pushing the Limits of Low-Resource Morphological Inflection*](https://arxiv.org/abs/1908.05838). EMNLP. 2019.


## Miscellaneous

- Environment (conda): `environment.yml`
- Pre-commit check: `pre-commit run --all-files`
- Compile: `make`


## License

MIT
