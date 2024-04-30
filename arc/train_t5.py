import torch

from transformers import AutoTokenizer

from loaders.single_loader import SingleLoader
from loaders.full_loader import FullLoader

from modeling.arc_longt5 import ArcLongT5
from training.t5_trainer import T5Trainer

import utils.constants as constants


MODEL_URL = 'google/long-t5-tglobal-base'

TRAIN_DATA_URL = 'cnn_dailymail'
VAL_DATA_URL = 'cnn_dailymail'

NAME = "arc-longt5"

TRAIN_CONFIG = {
    "lr": 0.001,
    "bs": 1,
    "accums": 128,
    "num_steps": 2000,
    "warmup_steps": 1,
    "eval_freq": 100,
    "checkpoint_freq": 500,
    "dtype": torch.bfloat16,
    "max_input_length": 2048,
    "max_output_length": 512,
    "max_eval_examples": 100
}


def main():
    
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_URL)

    model = ArcLongT5.from_pretrained(MODEL_URL)
    model.init_arc_head()
    model = model.to(constants.DEVICE)
    _ = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    print("Loading data...")
    train_loader = SingleLoader(TRAIN_DATA_URL, train=True, debug=False)
    val_loader = FullLoader(VAL_DATA_URL, train=False, debug=False)

    print("Train!")
    trainer = T5Trainer(
        NAME,
        **TRAIN_CONFIG
    )
    trainer.train(
        tokenizer,
        model,
        train_loader,
        val_loader
    )


if __name__ == "__main__":
        main()
