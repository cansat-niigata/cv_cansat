import chainer
import chainer.functions as F


class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.predictor = kwargs.pop('models')
        super().__init__(*args, **kwargs)

    def loss_predictor(self, predictor, gen_out, target):
        loss = F.mean_absolute_error(gen_out, target)
        chainer.report({'loss': loss}, predictor)
        return loss

    def forward(self, source, target):
        gen_out = self.predictor(source)
        loss = {
            'predictor': self.loss_predictor(
                predictor=self.predictor,
                gen_out=gen_out,
                target=target,
            )
        }
        return loss

    def update_core(self):
        opt_pre = self.get_optimizer('predictor')

        batch = self.get_iterator('main').next()
        batch = self.converter(batch, device=self.device)
        loss = self.forward(batch[0], batch[1])

        opt_pre.update(loss.get, 'predictor')


class Pix2PixUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.predictor, self.discriminator = kwargs.pop('models')
        super().__init__(*args, **kwargs)

    def loss_predictor(self, predictor, gen_out, target, dis_out, l1=100, l2=1):
        batchsize, _, h, w = dis_out.data.shape
        loss_mae = l1 * F.mean_absolute_error(gen_out, target)
        loss_adv = l2 * F.sum(F.softplus(-dis_out)) / batchsize / h / w
        loss = loss_mae + loss_adv

        chainer.report({'loss_mae': loss_mae}, predictor)
        chainer.report({'loss_adv': loss_adv}, predictor)
        return loss

    def loss_discriminator(self, discriminator, real, fake):
        batchsize, _, h, w = fake.data.shape
        loss1 = F.sum(F.softplus(-real)) / batchsize / h / w
        loss2 = F.sum(F.softplus(fake)) / batchsize / h / w
        loss = loss1 + loss2
        chainer.report({'loss': loss}, discriminator)
        return loss

    def forward(self, in_data, target):
        gen_out = self.predictor(in_data)
        dis_fake = self.discriminator(in_data, gen_out)
        dis_real = self.discriminator(in_data, target)

        loss = {
            'predictor': self.loss_predictor(
                predictor=self.predictor,
                gen_out=gen_out,
                target=target,
                dis_out=dis_fake
            ),
            'discriminator': self.loss_discriminator(
                discriminator=self.discriminator,
                real=dis_real,
                fake=dis_fake
            )
        }
        return loss

    def update_core(self):
        opt_pre = self.get_optimizer('predictor')
        opt_dis = self.get_optimizer('discriminator')

        batch = self.get_iterator('main').next()
        batch = self.converter(batch, device=self.device)
        loss = self.forward(batch[0], batch[1])

        opt_pre.update(loss.get, 'predictor')
        opt_dis.update(loss.get, 'discriminator')
