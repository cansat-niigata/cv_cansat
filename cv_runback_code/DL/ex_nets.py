import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain


class DWCBA(Chain):

    def __init__(self, in_ch, ch_multi, sample):
        super().__init__()
        with self.init_scope():
            if sample == 'down':
                self.dc = L.DepthwiseConvolution2D(in_ch, ch_multi, ksize=4, stride=2, pad=1)
                self.ac = F.leaky_relu
            elif sample == 'up':
                self.dc = L.Deconvolution2D(in_ch, in_ch * ch_multi, ksize=4, stride=2, pad=1)
                self.ac = F.relu
            else:
                self.dc = L.DepthwiseConvolution2D(in_ch, ch_multi, ksize=3, stride=1, pad=1)
                self.ac = None
            self.bn = L.BatchNormalization(in_ch * ch_multi)

    def __call__(self, x):
        h = self.dc(x)
        h = self.bn(h)
        if self.ac is not None:
            h = self.ac(h)
        return h


class PWCBA(Chain):

    def __init__(self, in_ch, out_ch, sample):
        super().__init__()
        with self.init_scope():
            if sample == 'down':
                self.ac = F.leaky_relu
            elif sample == 'up':
                self.ac = F.relu
            else:
                self.ac = None
            self.pc = L.Convolution2D(in_ch, out_ch, 1, 1, 0)
            self.bn = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = self.pc(x)
        h = self.bn(h)
        if self.ac is not None:
            h = self.ac(h)
        return h


class MobileNet(Chain):

    def __init__(self, in_ch, ch_multi, out_ch, sample):
        super().__init__()
        with self.init_scope():
            self.l0 = DWCBA(in_ch, ch_multi, sample)
            self.l1 = PWCBA(in_ch * ch_multi, out_ch, sample)

    def __call__(self, x):
        h = self.l0(x)
        h = self.l1(h)
        return h


class Encoder(Chain):

    def __init__(self, base=16):
        super().__init__()
        with self.init_scope():
            self.l0 = MobileNet(in_ch=3, ch_multi=base * 1, out_ch=base * 1, sample='same')
            self.l1 = MobileNet(in_ch=base * 1, ch_multi=2, out_ch=base * 2, sample='down')
            self.l2 = MobileNet(in_ch=base * 2, ch_multi=2, out_ch=base * 4, sample='down')
            self.l3 = MobileNet(in_ch=base * 4, ch_multi=2, out_ch=base * 4, sample='down')
            self.l4 = MobileNet(in_ch=base * 4, ch_multi=2, out_ch=base * 4, sample='down')
            self.l5 = MobileNet(in_ch=base * 4, ch_multi=2, out_ch=base * 4, sample='down')

    def __call__(self, x):
        h = [self.l0(x)]
        for i in range(1, 6):
            h.append(self['l%d' % i](h[i - 1]))
        return h


class Decoder(Chain):

    def __init__(self, base=16):
        w = chainer.initializers.Normal(0.02)
        super().__init__()
        with self.init_scope():
            self.l0 = MobileNet(in_ch=base * 4, ch_multi=2, out_ch=base * 4, sample='up')
            self.l1 = MobileNet(in_ch=base * 8, ch_multi=2, out_ch=base * 4, sample='up')
            self.l2 = MobileNet(in_ch=base * 8, ch_multi=2, out_ch=base * 4, sample='up')
            self.l3 = MobileNet(in_ch=base * 8, ch_multi=2, out_ch=base * 2, sample='up')
            self.l4 = MobileNet(in_ch=base * 4, ch_multi=2, out_ch=base * 1, sample='up')
            self.l5 = MobileNet(in_ch=base * 2, ch_multi=2, out_ch=1, sample='same')

    def __call__(self, x):
        h = self.l0(x[-1])
        for i in range(1, 6):
            h = F.concat([h, x[-i - 1]])
            h = self['l%d' % i](h)
        return h


class Predictor(Chain):

    def __init__(self, base=16):
        super().__init__()
        with self.init_scope():
            self.enc = Encoder(base=base)
            self.dec = Decoder(base=base)

    def __call__(self, x):
        return self.dec(self.enc(x))


class CBA(Chain):

    def __init__(self, in_ch, out_ch, sample):
        super().__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            if sample == 'down':
                self.c = L.Convolution2D(in_ch, out_ch, ksize=4, stride=2, pad=1)
            elif sample == 'up':
                self.c = L.Deconvolution2D(in_ch, out_ch, ksize=4, stride=2, pad=1)
            else:
                self.c = L.Convolution2D(in_ch, out_ch, ksize=1, stride=1, pad=0)

    def __call__(self, x):
        h = F.relu(self.bn(x))
        return self.c(h)


class Discriminator(Chain):

    def __init__(self, base=8):
        super().__init__()
        with self.init_scope():
            self.l0_0 = CBA(3, base * 1, sample='down')
            self.l0_1 = CBA(1, base * 1, sample='down')
            self.l1 = CBA(base * 2, base * 4, sample='down')
            self.l2 = CBA(base * 4, base * 8, sample='down')
            self.l3 = CBA(base * 8, base * 8, sample='down')
            self.l4 = CBA(base * 8, base * 8, sample='down')

    def __call__(self, x, y):
        h0 = self.l0_0(x)
        h1 = self.l0_1(y)
        h = F.concat((h0, h1))
        h = self.l1(h)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        return h