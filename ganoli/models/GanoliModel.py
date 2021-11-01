import sys

sys.path.append('/om2/user/rogerjin/GANOLI/ganoli')
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from os.path import join as opj
import numpy as np
from GanoliDataset import GanoliUnimodalDataset, GanoliMultimodalDataset

seed_everything(42, workers=True)


class GanoliGAN(pl.LightningModule):

    def __init__(self, generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac):
        super().__init__()
        self.automatic_optimization = False

        self.generator_rna2atac = generator_rna2atac
        self.generator_atac2rna = generator_atac2rna
        self.discriminator_rna = discriminator_rna
        self.discriminator_atac = discriminator_atac

        self.generator_loss_fn = MSELoss()
        self.discriminator_loss_fn = CrossEntropyLoss()

        self.supervised = False

    def supervise(self):
        self.supervised = True

    def unsupervise(self):
        self.supervised = False

    def forward(self, rna_or_atac, data_type):
        if data_type == 'rna':
            return self.generator_rna2atac(rna_or_atac)

        if data_type == 'atac':
            return self.generator_atac2rna(rna_or_atac)

        else:
            raise ValueError(f"Invalid datatype {data_type}, expected 'rna' or 'atac'")


    def training_step(self, batch, batch_idx):

        if self.supervised:
            return self.supervised_training_step(batch, batch_idx)

        else:
            return self.unsupervised_training_step(batch, batch_idx)


    def supervised_training_step(self, batch, batch_idx):

        print(batch_idx)

        rna_real, atac_real = batch['rna'], batch['atac']

        for opt in self.optimizers():
            opt.zero_grad()

        atac_generated = self.generator_rna2atac(rna_real)
        rna_generated = self.generator_atac2rna(atac_real)

        loss = self.generator_loss(atac_generated, atac_real) + self.generator_loss(rna_generated, rna_real)
        self.manual_backward(loss)

        for opt in self.optimizers():
            opt.step()

        return loss

    def unsupervised_training_step(self, batch, batch_idx):

        print(batch_idx)

        for opt in self.optimizers():
            opt.zero_grad()

        rna_real, atac_real = batch['rna'], batch['atac']
        rna_fake, atac_fake = self.generator_atac2rna(atac_real), self.generator_rna2atac(rna_real)
        rna_recon, atac_recon = self.generator_atac2rna(atac_fake), self.generator_rna2atac(rna_real)
        generator_loss = self.generator_loss_fn(rna_real, rna_recon) + self.generator_loss_fn(atac_real, atac_recon)

        discr_rna_real, discr_atac_real = self.discriminator_rna(rna_real), self.discriminator_atac(atac_real)
        discr_rna_fake, discr_atac_fake = self.discriminator_rna(rna_fake), self.discriminator_atac(atac_fake)
        discriminator_loss_real = self.discriminator_loss_fn(discr_rna_real, torch.ones_like(discr_rna_real)) + self.discriminator_loss_fn(discr_atac_real, torch.ones_like(discr_atac_real))
        discriminator_loss_fake = self.discriminator_loss_fn(discr_rna_fake, torch.zeros_like(discr_rna_fake)) + self.discriminator_loss_fn(discr_atac_fake, torch.ones_like(discr_atac_fake))

        loss = generator_loss + discriminator_loss_real + discriminator_loss_fake
        self.manual_backward(loss)

        for opt in self.optimizers():
            opt.step()

        return loss

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
        self.input_modality = input_modality


    def forward(self, inp):
        return self.model(inp)


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
        discriminator_atac = GanoliLinearDiscriminator(rna_shape, input_modality='atac')
        super().__init__(generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac)


if __name__ == '__main__':
    data_root = '/om2/user/rogerjin/data'
    data_path = opj(data_root, 'data_files_new.npz')
    data = np.load(data_path)

    gan = GanoliLinearGAN(7445, 3808)

    trainer = Trainer(gpus=1)
    train_rna = GanoliUnimodalDataset(data['rna_train'])
    train_atac = GanoliUnimodalDataset(data['atac_train_small'])
    rna_atac = GanoliMultimodalDataset(rna=train_rna, atac=train_atac)

    def collate_fn(batch):
        rna, atac = batch['rna'], batch['atac']
        return {'rna': rna.float(), 'atac': atac.float()}

    train_dataloader = DataLoader(rna_atac, collate_fn=collate_fn)
    trainer.fit(gan, train_dataloader)
