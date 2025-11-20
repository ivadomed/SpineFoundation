
import argparse
from .trainer import Trainer

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--model-params',type=str,required=True)

    p.add_argument('--data-params',type=str,required=True)

    p.add_argument('--training-params',type=str,required=True)

    return p.parse_args()


def train():
    args = parse_args()
    trainer = Trainer(args)
    trainer.fit()


if __name__ == '__main__':
    train()