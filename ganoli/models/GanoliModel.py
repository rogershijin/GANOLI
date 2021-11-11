import sys

sys.path.append('/om2/user/rogerjin/GANOLI/ganoli')

from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)
from torch.nn import MSELoss, BCEWithLogitsLoss
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
        self.generator_loss_fn = BCEWithLogitsLoss()
        self.discriminator_loss_fn = BCEWithLogitsLoss()

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
        discriminator_loss_real = self.discriminator_loss_fn(real_preds,
                                                             torch.ones(real_preds.shape[0], 1).to(self.device))
        discriminator_loss_fake = self.discriminator_loss_fn(fake_preds,
                                                             torch.zeros(fake_preds.shape[0], 1).to(self.device))
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
            return self.unsupervised_step(batch, batch_idx, optimizer_idx=optimizer_idx, data_partition='train')

    def validation_step(self, batch, batch_idx):

        if not self.supervised:
            return self.unsupervised_step(batch, batch_idx, data_partition='validation')

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

    def unsupervised_step(self, batch, batch_idx, optimizer_idx=None, data_partition='train'):

        print(data_partition)

        rna_real, atac_real = batch['rna'].float(), batch['atac'].float()
        rna_fake, atac_fake = self.generator_atac2rna(atac_real), self.generator_rna2atac(rna_real)

        if data_partition == 'validation' or optimizer_idx in [0, 1]:  # generator

            rna_recon, atac_recon = self.generator_atac2rna(atac_fake), self.generator_rna2atac(rna_fake)

            rna_recon_loss = self.reconstruction_loss(rna_real, rna_recon)
            atac_recon_loss = self.reconstruction_loss(atac_real, atac_recon)
            total_recon_loss = rna_recon_loss + atac_recon_loss

            self.log(f'{data_partition}_loss/recon_rna', rna_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{data_partition}_loss/recon_atac', atac_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{data_partition}_loss/recon_total', total_recon_loss, on_step=False, on_epoch=True,
                     prog_bar=True)

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

            self.log(f'{data_partition}_loss/gen_atac2rna', atac2rna_gen_loss, on_step=False, on_epoch=True,
                     prog_bar=True)
            self.log(f'{data_partition}_loss/gen_rna2atac', rna2atac_gen_loss, on_step=False, on_epoch=True,
                     prog_bar=True)
            self.log(f'{data_partition}_loss/gen_total', total_gen_loss, on_step=False, on_epoch=True, prog_bar=True)

            rna_oracle_recon_loss = self.reconstruction_loss(rna_fake, rna_real)
            atac_oracle_recon_loss = self.reconstruction_loss(atac_fake, atac_real)
            total_oracle_recon_loss = rna_oracle_recon_loss + atac_oracle_recon_loss

            self.log(f'{data_partition}_oracle/oracle_rna', rna_oracle_recon_loss, on_step=False, on_epoch=True,
                     prog_bar=True)
            self.log(f'{data_partition}_oracle/oracle_atac', atac_oracle_recon_loss, on_step=False, on_epoch=True,
                     prog_bar=True)
            self.log(f'{data_partition}_oracle/oracle_total', total_oracle_recon_loss, on_step=False, on_epoch=True,
                     prog_bar=True)

            if data_partition == 'validation':
                self.log(f'checkpointer_objective', total_oracle_recon_loss, on_step=False, on_epoch=True)

            # return total_recon_loss + total_id_loss + total_gen_loss
            if data_partition == 'train':
                return total_recon_loss + total_gen_loss

        if data_partition == 'validation' or optimizer_idx in [2, 3]:  # discriminator
            discr_rna_real, discr_atac_real = self.discriminator_rna(rna_real), self.discriminator_atac(atac_real)
            discr_rna_fake, discr_atac_fake = self.discriminator_rna(rna_fake), self.discriminator_atac(atac_fake)

            rna_discr_loss = self.discriminator_loss(discr_rna_real, discr_rna_fake)
            atac_discr_loss = self.discriminator_loss(discr_atac_real, discr_atac_fake)
            total_discr_loss = rna_discr_loss + atac_discr_loss

            self.log(f'{data_partition}_loss/discr_rna', rna_discr_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{data_partition}_loss/discr_atac', atac_discr_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{data_partition}_loss/discr_total', total_discr_loss, on_step=False, on_epoch=True,
                     prog_bar=True)

            if data_partition == 'train':
                return total_discr_loss

        return total_recon_loss + total_gen_loss + total_discr_loss

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


class GanoliShallowGenerator(GanoliGenerator):

    def __init__(self, input_shape, output_shape, input_modality='atac', hidden_dim=500, bias=True):
        super().__init__(input_modality=input_modality)
        self.linear1 = nn.Linear(input_shape, hidden_dim, bias=bias)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_dim, output_shape, bias=bias)

        def model(x):
            x = self.linear1(x)
            x = self.leaky_relu(x)
            x = self.linear2(x)
            return x

        self.model = model


class GanoliShallowDiscriminator(GanoliDiscriminator):
    def __init__(self, input_shape, input_modality='atac', hidden_dim=500, bias=True):
        super().__init__(input_modality=input_modality)

        self.linear1 = nn.Linear(input_shape, hidden_dim, bias=bias)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_dim, 1, bias=bias)

        def model(x):
            x = self.linear1(x)
            x = self.leaky_relu(x)
            x = self.linear2(x)
            return x

        self.model = model


class GanoliShallowGAN(GanoliGAN):
    def __init__(self, rna_shape, atac_shape, hidden_dim=500, bias=True):
        generator_rna2atac = GanoliShallowGenerator(rna_shape, atac_shape, input_modality='rna', hidden_dim=hidden_dim,
                                                    bias=bias)
        generator_atac2rna = GanoliShallowGenerator(atac_shape, rna_shape, input_modality='atac', hidden_dim=hidden_dim,
                                                    bias=bias)
        discriminator_rna = GanoliShallowDiscriminator(rna_shape, input_modality='rna', hidden_dim=hidden_dim,
                                                       bias=bias)
        discriminator_atac = GanoliShallowDiscriminator(atac_shape, input_modality='atac', hidden_dim=hidden_dim,
                                                        bias=bias)
        super().__init__(generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac)


if __name__ == '__main__':
    data_root = '/om2/user/rogerjin/data'
    data_path = opj(data_root, 'data_files_new.npz')
    data = np.load(data_path)

    # gan = GanoliLinearGAN(7445, 3808)
    gan = GanoliShallowGAN(7445, 3808)

    kwargs = {}
    if torch.cuda.is_available():
        kwargs['gpus'] = -1

    tb_logger = loggers.TensorBoardLogger("logs/debug/")
    checkpointer = ModelCheckpoint(monitor='checkpointer_objective',
                                   filename='epoch={epoch:02d}-val_oracle_total={checkpointer_objective:.2f}',
                                   save_top_k=10, auto_insert_metric_name=False)

    trainer = Trainer(**kwargs, logger=tb_logger, callbacks=[checkpointer])
    train_rna = GanoliUnimodalDataset(data['rna_train'])
    train_atac = GanoliUnimodalDataset(data['atac_train_small'])
    train_rna_atac = GanoliMultimodalDataset(rna=train_rna, atac=train_atac)

    val_rna = GanoliUnimodalDataset(data['rna_test'])
    val_atac = GanoliUnimodalDataset(data['atac_test_small'])
    val_rna_atac = GanoliMultimodalDataset(rna=val_rna, atac=val_atac)

    train_dataloader = DataLoader(train_rna_atac, batch_size=32, num_workers=4)
    val_dataloader = DataLoader(val_rna_atac, batch_size=32, num_workers=4)
    trainer.fit(gan, train_dataloader, val_dataloader)
