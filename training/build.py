import argparse

import training.train

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--model-name',type=str,default='spine_mae')

    p.add_argument('--model-params',type=str,default='',help='JSON string of model constructor params to override defaults')

    #model-params prend le dessus sur ces paramètres

    p.add_argument('--img-size', nargs=3, type=int, default=(32, 32, 32))
    p.add_argument('--patch-size', nargs=3, type=int, default=(8, 8, 8))
    p.add_argument('--in-channels', type=int, default=1)
    p.add_argument('--embed-dim', type=int, default=64)
    p.add_argument('--enc-layers', type=int, default=2)
    p.add_argument('--enc-mlp-dim', type=int, default=128)
    p.add_argument('--num-heads', type=int, default=4)

    p.add_argument('--dec-embed-dim', type=int, default=32)
    p.add_argument('--dec-layers', type=int, default=1)
    p.add_argument('--dec-num-heads', type=int, default=4)
    p.add_argument('--dec-mlp-dim', type=int, default=3072)

    # Training
    p.add_argument('--data-params', type=str, default='',
               help='JSON override for training data parameters')

    p.add_argument('--train-ratio', type=float, default=0.8)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--test-ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=28)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--data_path', type=str, default='')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--mask-ratio', type=float, default=0.5)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--no-cuda', action='store_true')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--work-dir', type=str, default='./training/ckpt')
    p.add_argument('--num-workers', type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()
    args.img_size = tuple(map(int, args.img_size))
    args.patch_size = tuple(map(int, args.patch_size))
    os.makedirs(args.work_dir, exist_ok=True)

    trainer = Trainer(args)
    trainer.fit()


if __name__ == '__main__':
    main()