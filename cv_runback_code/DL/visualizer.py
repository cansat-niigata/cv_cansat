import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable


def out_image(updater, pre, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        xp = pre.xp

        w_in = 256
        w_out = 256

        batch = updater.get_iterator('test').next()
        source = xp.asarray(batch[0][0]).reshape(1, 3, 256, 256)
        target = xp.asarray(batch[0][1])
        gen_out = pre(Variable(source))

        source = (source + 1) * 128
        target = (target + 1) * 128
        Max = xp.max(gen_out.data)
        if Max == 0:
            Max += 1e-8
        gen_out = gen_out.data / Max * 255

        source = xp.asnumpy(source)
        target = xp.asnumpy(target)
        gen_out = xp.asnumpy(gen_out)

        source = source.reshape(3, w_in, w_in).transpose((1, 2, 0)).astype(np.uint8)
        target = target.reshape(w_out, w_out).astype(np.uint8)
        gen_out = gen_out.reshape(w_out, w_out).astype(np.uint8)

        Image.fromarray(source).convert('RGB').save('./{}/source_{}.png'.format(dst, trainer.updater.iteration))
        Image.fromarray(target).convert('L').save('./{}/target_{}.png'.format(dst, trainer.updater.iteration))
        Image.fromarray(gen_out).convert('L').save('./{}/gen_out_{}.png'.format(dst, trainer.updater.iteration))

    return make_image