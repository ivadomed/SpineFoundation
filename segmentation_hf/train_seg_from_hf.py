from .config import parse_args
from .trainer import train


if __name__ == "__main__":
    config = parse_args()
    train(config)
