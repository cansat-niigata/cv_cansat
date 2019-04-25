import chainer
from chainer import training
from chainer.training import extensions
from chainer.datasets import TupleDataset

from ex_nets import Predictor, Discriminator
# from nets import Predictor, Discriminator
from updater import Pix2PixUpdater as Updater
from dataset import make_dataset
from visualizer import out_image

import argparse
import pickle
import cupy as cp


def main():
    parser = argparse.ArgumentParser(description='this network get red corn mask image')
    parser.add_argument('--batchsize', '-b', type=int, default=1)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--source_dir', default='./source')
    parser.add_argument('--target_dir', default='./target')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out', default='result')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', default='')
    parser.add_argument('--snapshot_interval', type=int, default=1)
    parser.add_argument('--display_interval', type=int, default=100)
    args = parser.parse_args()

    # Setup a neural network to train
    predictor = Predictor(base=32)
    discriminator = Discriminator(base=8)
    # predictor = Predictor(in_ch=3, out_ch=1, base=16)
    predictor.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=2e-4, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')
        return optimizer
    # def make_optimizer(model):
    #     optimizer = chainer.optimizers.MomentumSGD()
    #     optimizer.setup(model)
    #     optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_dec')
    #     return optimizer

    opt_pre = make_optimizer(predictor)
    opt_dis = make_optimizer(discriminator)

    # source, target = make_dataset(args.source_dir, args.target_dir)
    with open('source.pickle', 'rb') as f:
        source = pickle.load(f)
    with open('target.pickle', 'rb') as f:
        target = pickle.load(f)
    source = cp.array(source, dtype='f') / 128 - 1
    target = cp.array(target, dtype='f') / 128 - 1
    dataset = TupleDataset(source, target)
    train_data = dataset
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

    # Setup a trainer
    updater = Updater(
        models=(predictor, discriminator),
        iterator={'main': train_iter, 'test': test_iter},
        optimizer={'predictor': opt_pre, 'discriminator': opt_dis},
        device=args.gpu
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'epoch')
    display_interval = (args.display_interval, 'iteration')

    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(predictor, 'pre_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'predictor/loss_mae', 'predictor/loss_adv', 'discriminator/loss'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(out_image(updater, predictor, args.seed, 'result/sample'), trigger=snapshot_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, predictor)

    trainer.run()

    # if args.gpu >= 0:
    #     predictor.to_cpu()

    chainer.serializers.save_npz('predictor', predictor)


if __name__ == '__main__':
    main()
