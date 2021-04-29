"""
test
"""
from decoding import get_decode_fn
from train import Trainer


def main():
    """
    main
    """
    trainer = Trainer()
    params = trainer.params
    decode_fn = get_decode_fn(
        params.decode, params.max_decode_len, params.decode_beam_size
    )
    trainer.load_data(params.dataset, params.train, params.dev, params.test)
    trainer.setup_evalutator()

    assert params.load
    trainer.reload_and_test(params.model, params.load, params.bs, decode_fn)


if __name__ == "__main__":
    main()
