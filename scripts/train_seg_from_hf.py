from segmentation_hf.config import parse_args
from segmentation_hf.trainer import train


if __name__ == "__main__":
    config = parse_args()
    train(config)
