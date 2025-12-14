import argparse
import os
from .runner import InferenceRunner

def build_inference():
    pass  # Implementation of build_inference goes here

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--config', type=str, required=True)

    p.add_argument('--model-ckpt', type=str, required=True)


    p.add_argument('--outdir', type=str, required=True)


    return p.parse_args()


def main():
    args = parse_args()
    runner = InferenceRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
