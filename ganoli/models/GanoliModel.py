import sys

sys.path.append('/om2/user/rogerjin/GANOLI/ganoli')

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, loggers
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
from torch.nn import MSELoss, BCELoss
from torch.utils.data import DataLoader
from os.path import join as opj
import numpy as np
from GanoliDataset import GanoliUnimodalDataset, GanoliMultimodalDataset

seed_everything(42, workers=True)


class GanoliGAN(pl.LightningModule):

    def __init__(self, generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac):
        super().__init__()
        # self.automatic_optimization = False

        self.generator_rna2atac = generator_rna2atac
        self.generator_atac2rna = generator_atac2rna
        self.discriminator_rna = discriminator_rna
        self.discriminator_atac = discriminator_atac
        self.supervised = False

        self.reconstruction_loss_fn = MSELoss()
        self.identity_loss_fn = MSELoss()
        self.generator_loss_fn = BCELoss()
        self.discriminator_loss_fn = BCELoss()

    def supervise(self):
        self.supervised = True

    def unsupervise(self):
        self.supervised = False

    def reconstruction_loss(self, pred, target):
        return self.reconstruction_loss_fn(pred, target)

    def identity_loss(self, pred, target):
        return self.identity_loss_fn(pred, target)

    def generator_loss(self, fake_preds):
        return self.generator_loss_fn(fake_preds, torch.ones(fake_preds.shape[0], 1).to(self.device))

    def discriminator_loss(self, real_preds, fake_preds):
        discriminator_loss_real = self.discriminator_loss_fn(real_preds, torch.ones(real_preds.shape[0], 1).to(self.device))
        discriminator_loss_fake = self.discriminator_loss_fn(fake_preds, torch.zeros(fake_preds.shape[0], 1).to(self.device))
        return discriminator_loss_real + discriminator_loss_fake

    def forward(self, rna_or_atac, data_type):
        if data_type == 'rna':
            return self.generator_rna2atac(rna_or_atac)

        if data_type == 'atac':
            return self.generator_atac2rna(rna_or_atac)

        else:
            raise ValueError(f"Invalid datatype {data_type}, expected 'rna' or 'atac'")

    def on_epoch_start(self):
        print('\n')


    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.supervised:
            return self.supervised_training_step(batch, batch_idx)

        else:
            return self.unsupervised_training_step(batch, batch_idx, optimizer_idx)


    def supervised_training_step(self, batch, batch_idx):

        rna_real, atac_real = batch['rna'].float(), batch['atac'].float()

        for opt in self.optimizers():
            opt.zero_grad()

        atac_generated = self.generator_rna2atac(rna_real)
        rna_generated = self.generator_atac2rna(atac_real)
        loss = self.reconstruction_loss(atac_generated, atac_real) + self.reconstruction_loss(rna_generated, rna_real)

        self.log('supervised_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.manual_backward(loss)

        for opt in self.optimizers():
            opt.step()

        return loss

    def unsupervised_training_step(self, batch, batch_idx, optimizer_idx):

        rna_real, atac_real = batch['rna'].float(), batch['atac'].float()
        rna_fake, atac_fake = self.generator_atac2rna(atac_real), self.generator_rna2atac(rna_real)

        if optimizer_idx in [0,1]: # generator

            rna_recon, atac_recon = self.generator_atac2rna(atac_fake), self.generator_rna2atac(rna_real)

            rna_recon_loss = self.reconstruction_loss(rna_real, rna_recon)
            atac_recon_loss = self.reconstruction_loss(atac_real, atac_recon)
            total_recon_loss = rna_recon_loss + atac_recon_loss

            self.log('loss/recon_rna', rna_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('loss/recon_atac', atac_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('loss/recon_total', total_recon_loss, on_step=False, on_epoch=True, prog_bar=True)

            # todo: think about identity loss - not easy because atac/rna come in different sizes.

            # rna_id, atac_id = self.generator_atac2rna(rna_real), self.generator_rna2atac(atac_real)

            # rna_id_loss = self.identity_loss(rna_id, rna_real)
            # atac_id_loss = self.identity_loss(atac_id, atac_real)
            # total_id_loss = rna_id_loss + atac_id_loss

            # self.log('loss/id_rna', rna_id_loss, on_step=False, on_epoch=True, prog_bar=True)
            # self.log('loss/id_atac', atac_id_loss, on_step=False, on_epoch=True, prog_bar=True)
            # self.log('loss/id_total', total_id_loss, on_step=False, on_epoch=True, prog_bar=True)

            discr_rna_fake, discr_atac_fake = self.discriminator_rna(rna_fake), self.discriminator_atac(atac_fake)

            atac2rna_gen_loss = self.generator_loss(discr_rna_fake)
            rna2atac_gen_loss = self.generator_loss(discr_atac_fake)
            total_gen_loss = atac2rna_gen_loss + rna2atac_gen_loss

            self.log('loss/atac2rna_generator_loss', atac2rna_gen_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('loss/rna2atac_generator_loss', rna2atac_gen_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('loss/total_generator_loss', total_gen_loss, on_step=False, on_epoch=True, prog_bar=True)

            # return total_recon_loss + total_id_loss + total_gen_loss
            return total_recon_loss + total_gen_loss

        elif optimizer_idx in [2,3]: # discriminator
            discr_rna_real, discr_atac_real = self.discriminator_rna(rna_real), self.discriminator_atac(atac_real)
            discr_rna_fake, discr_atac_fake = self.discriminator_rna(rna_fake), self.discriminator_atac(atac_fake)

            rna_discr_loss = self.discriminator_loss(discr_rna_real, discr_rna_fake)
            atac_discr_loss = self.discriminator_loss(discr_atac_real, discr_atac_fake)
            total_discr_loss = rna_discr_loss + atac_discr_loss

            self.log('loss/rna_discriminator_loss', rna_discr_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('loss/atac_discriminator_loss', atac_discr_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('loss/total_discriminator_loss', total_discr_loss, on_step=False, on_epoch=True, prog_bar=True)

            return total_discr_loss


    def configure_optimizers(self):

        generator_rna2atac_opt = torch.optim.Adam(self.generator_rna2atac.parameters())
        generator_atac2rna_opt = torch.optim.Adam(self.generator_atac2rna.parameters())

        generator_opts = [generator_rna2atac_opt, generator_atac2rna_opt]

        discriminator_opts = []

        if not self.supervised:
            discriminator_rna_opt = torch.optim.Adam(self.discriminator_rna.parameters())
            discriminator_atac_opt = torch.optim.Adam(self.discriminator_atac.parameters())
            discriminator_opts = [discriminator_rna_opt, discriminator_atac_opt]

        opts = generator_opts + discriminator_opts

        return opts


class GanoliGenerator(pl.LightningModule):

    def __init__(self, input_modality='atac', output_modality='rna'):
        super().__init__()

        self.model = None
        self.input_modality = input_modality
        self.output_modality = output_modality


    def forward(self, inp):
        return self.model(inp)


    def __str__(self):
        return f'Generator {self.input_modality} --> {self.output_modality}'


class GanoliDiscriminator(pl.LightningModule):

    def __init__(self, input_modality='atac'):
        super().__init__()

        self.model = None
        self.sigmoid = nn.Sigmoid()
        self.input_modality = input_modality


    def forward(self, inp):
        x = self.model(inp)
        return self.sigmoid(x)


class GanoliLinearGenerator(GanoliGenerator):

    def __init__(self, input_shape, output_shape, input_modality='atac', bias=True):
        super().__init__(input_modality=input_modality)

        self.model = nn.Linear(input_shape, output_shape, bias=bias)


class GanoliLinearDiscriminator(GanoliDiscriminator):

    def __init__(self, input_shape, input_modality='atac', bias=True):
        super().__init__(input_modality=input_modality)

        self.model = nn.Linear(input_shape, 1, bias=bias)


class GanoliLinearGAN(GanoliGAN):

    def __init__(self, rna_shape, atac_shape, bias=True):
        generator_rna2atac = GanoliLinearGenerator(rna_shape, atac_shape, input_modality='rna')
        generator_atac2rna = GanoliLinearGenerator(atac_shape, rna_shape, input_modality='atac')
        discriminator_rna = GanoliLinearDiscriminator(rna_shape, input_modality='rna')
        discriminator_atac = GanoliLinearDiscriminator(atac_shape, input_modality='atac')
        super().__init__(generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac)


if __name__ == '__main__':
    data_root = '/om2/user/rogerjin/data'
    data_path = opj(data_root, 'data_files_new.npz')
    data = np.load(data_path)

    gan = GanoliLinearGAN(7445, 3808)

    kwargs = {}
    if torch.cuda.is_available():
        kwargs['gpus'] = -1

    tb_logger = loggers.TensorBoardLogger("linear/")

    trainer = Trainer(**kwargs, logger=tb_logger)
    train_rna = GanoliUnimodalDataset(data['rna_train'])
    train_atac = GanoliUnimodalDataset(data['atac_train_small'])
    rna_atac = GanoliMultimodalDataset(rna=train_rna, atac=train_atac)

    train_dataloader = DataLoader(rna_atac, batch_size=32, num_workers=4)
    trainer.fit(gan, train_dataloader)
