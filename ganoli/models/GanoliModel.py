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
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import anndata as ad

seed_everything(42, workers=True)


class GanoliGAN(pl.LightningModule):

    def __init__(self, generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac, hypers={}):
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

        self.hypers = hypers
        self.log_hyperparams(self.hypers)

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

    def on_epoch_end(self):
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
                self.log(f'checkpointer_objective', rna_oracle_recon_loss, on_step=False, on_epoch=True)

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

        lr = self.hypers.get('lr', 0.0002)

        generator_rna2atac_opt = torch.optim.Adam(self.generator_rna2atac.parameters(), lr=lr, betas=[0.5, 0.999])
        generator_atac2rna_opt = torch.optim.Adam(self.generator_atac2rna.parameters(), lr=lr, betas=[0.5, 0.999])

        generator_opts = [generator_rna2atac_opt, generator_atac2rna_opt]

        discriminator_opts = []

        if not self.supervised:
            discriminator_rna_opt = torch.optim.Adam(self.discriminator_rna.parameters(), lr=lr, betas=[0.5, 0.999])
            discriminator_atac_opt = torch.optim.Adam(self.discriminator_atac.parameters(), lr=lr, betas=[0.5, 0.999])
            discriminator_opts = [discriminator_rna_opt, discriminator_atac_opt]

        opts = generator_opts + discriminator_opts

        return opts


class GanoliGenerator(pl.LightningModule):

    def __init__(self, input_modality='atac', output_modality='rna', embedding=None, embedding_labels=None):
        super().__init__()

        # self.tensorboard = self.logger.experiment

        self.model = None
        self.input_modality = input_modality
        self.output_modality = output_modality
        self.embedding=embedding
        if self.embedding is not None:
            self.embedding.to(self.device)
            if embedding_labels is not None:
                self.tensorboard.add_embedding(embedding, metadata=embedding_labels, tag=f'generator_{self.input_modality}_embedding')

    def forward(self, inp):
        if self.embedding is not None:
            inp = inp @ self.embedding
        return self.model(inp)

    def __str__(self):
        return f'Generator {self.input_modality} --> {self.output_modality}'


class GanoliDiscriminator(pl.LightningModule):

    def __init__(self, input_modality='atac', embedding=None, embedding_labels=None):
        super().__init__()

        self.model = None
        self.input_modality = input_modality
        self.embedding=embedding
        if self.embedding is not None:
            self.embedding.to(self.device)
            if embedding_labels is not None:
                self.tensorboard.add_embedding(embedding, metadata=embedding_labels, tag=f'discriminator_{self.input_modality}_embedding')

    def forward(self, inp):
        if self.embedding is not None:
            inp = inp @ self.embedding
        x = self.model(inp)
        return x


class GanoliLinearGenerator(GanoliGenerator):

    def __init__(self, input_shape, output_shape, input_modality='atac', bias=True, **kwargs):
        super().__init__(input_modality=input_modality, **kwargs)

        self.model = nn.Linear(input_shape, output_shape, bias=bias)

class GanoliLogisticGenerator(GanoliGenerator):

    def __init__(self, input_shape, output_shape, input_modality='atac', bias=True, **kwargs):
        super().__init__(input_modality=input_modality, **kwargs)
        self.linear = nn.Linear(input_shape, output_shape, bias=bias)
        self.sigmoid = nn.Sigmoid()

        def model(inp):
            return self.sigmoid(self.linear(inp))

        self.model = model


class GanoliLinearDiscriminator(GanoliDiscriminator):

    def __init__(self, input_shape, input_modality='atac', bias=True, **kwargs):
        super().__init__(input_modality=input_modality, **kwargs)

        self.model = nn.Linear(input_shape, 1, bias=bias)


class GanoliLinearGAN(GanoliGAN):

    def __init__(self, rna_shape, atac_shape, bias=True, rna_embedding=None, atac_embedding=None):
        generator_rna2atac = GanoliLinearGenerator(rna_shape, atac_shape, input_modality='rna', embedding=rna_embedding)
        generator_atac2rna = GanoliLinearGenerator(atac_shape, rna_shape, input_modality='atac', embedding=atac_embedding)
        discriminator_rna = GanoliLinearDiscriminator(rna_shape, input_modality='rna', embedding=rna_embedding)
        discriminator_atac = GanoliLinearDiscriminator(atac_shape, input_modality='atac', embedding=atac_embedding)
        super().__init__(generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac)

class GanoliLogisticGAN(GanoliGAN):

    def __init__(self, rna_shape, atac_shape, bias=True, rna_embedding=None, atac_embedding=None):
        generator_rna2atac = GanoliLogisticGenerator(rna_shape, atac_shape, input_modality='rna', embedding=rna_embedding)
        generator_atac2rna = GanoliLinearGenerator(atac_shape, rna_shape, input_modality='atac', embedding=atac_embedding)
        discriminator_rna = GanoliLinearDiscriminator(rna_shape, input_modality='rna', embedding=rna_embedding)
        discriminator_atac = GanoliLinearDiscriminator(atac_shape, input_modality='atac', embedding=atac_embedding)
        super().__init__(generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac)


class GanoliShallowGenerator(GanoliGenerator):

    def __init__(self, input_shape, output_shape, input_modality='atac', hidden_dim=500, bias=True, **kwargs):
        super().__init__(input_modality=input_modality, **kwargs)
        self.linear1 = nn.Linear(input_shape, hidden_dim, bias=bias)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_dim, output_shape, bias=bias)

        def model(x):
            x = self.linear1(x)
            x = self.leaky_relu(x)
            x = self.linear2(x)
            return x

        self.model = model
        
class GanoliShallowLogisticGenerator(GanoliGenerator):

    def __init__(self, input_shape, output_shape, input_modality='atac', hidden_dim=500, bias=True, **kwargs):
        super().__init__(input_modality=input_modality, **kwargs)
        self.linear1 = nn.Linear(input_shape, hidden_dim, bias=bias)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_dim, output_shape, bias=bias)
        self.sigmoid = nn.Sigmoid()

        def model(x):
            x = self.linear1(x)
            x = self.leaky_relu(x)
            x = self.linear2(x)
            x = self.sigmoid(x)
            return x

        self.model = model


class GanoliShallowDiscriminator(GanoliDiscriminator):
    def __init__(self, input_shape, input_modality='atac', hidden_dim=500, bias=True, **kwargs):
        super().__init__(input_modality=input_modality, **kwargs)

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
    def __init__(self, rna_shape, atac_shape, hidden_dim=500, bias=True, rna_embedding=None, atac_embedding=None, rna_embedding_labels=None):
        generator_rna2atac = GanoliShallowGenerator(rna_shape, atac_shape, input_modality='rna', hidden_dim=hidden_dim,
                                                    bias=bias, embedding=rna_embedding, embedding_labels=rna_embedding_labels)
        generator_atac2rna = GanoliShallowGenerator(atac_shape, rna_shape, input_modality='atac', hidden_dim=hidden_dim,
                                                    bias=bias, embedding=atac_embedding)
        discriminator_rna = GanoliShallowDiscriminator(rna_shape, input_modality='rna', hidden_dim=hidden_dim,
                                                       bias=bias, embedding=rna_embedding, embedding_labels=rna_embedding_labels)
        discriminator_atac = GanoliShallowDiscriminator(atac_shape, input_modality='atac', hidden_dim=hidden_dim,
                                                        bias=bias, embedding=atac_embedding)
        super().__init__(generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac)
        
        


class GanoliPCAGAN(GanoliGAN):
    def __init__(self, rna_shape, atac_shape, bias=True, rna_embedding=None, atac_embedding=None, rna_embedding_labels=None, pca_n_components=20):
        generator_rna2atac = GanoliLogisticGenerator(pca_n_components, atac_shape, input_modality='rna',
                                                    bias=bias, embedding=rna_embedding, embedding_labels=rna_embedding_labels)
        generator_atac2rna = GanoliLinearGenerator(pca_n_components, rna_shape, input_modality='atac',
                                                    bias=bias, embedding=atac_embedding)
        discriminator_rna = GanoliLinearDiscriminator(pca_n_components, input_modality='rna',
                                                       bias=bias, embedding=rna_embedding, embedding_labels=rna_embedding_labels)
        discriminator_atac = GanoliLinearDiscriminator(pca_n_components, input_modality='atac',
                                                        bias=bias, embedding=atac_embedding)
        super().__init__(generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac)
        
class GanoliShallowLogisticPCAGAN(GanoliGAN):
    def __init__(self, rna_shape, atac_shape, hidden_dim=500, bias=True, rna_embedding=None, atac_embedding=None, rna_embedding_labels=None, pca_n_components=20, hypers={}):
        generator_rna2atac = GanoliShallowLogisticGenerator(pca_n_components, atac_shape, input_modality='rna', hidden_dim=hidden_dim,
                                                    bias=bias, embedding=rna_embedding, embedding_labels=rna_embedding_labels)
        generator_atac2rna = GanoliShallowGenerator(pca_n_components, rna_shape, input_modality='atac', hidden_dim=hidden_dim,
                                                    bias=bias, embedding=atac_embedding)
        discriminator_rna = GanoliShallowDiscriminator(pca_n_components, input_modality='rna', hidden_dim=hidden_dim,
                                                       bias=bias, embedding=rna_embedding, embedding_labels=rna_embedding_labels)
        discriminator_atac = GanoliShallowDiscriminator(pca_n_components, input_modality='atac', hidden_dim=hidden_dim,
                                                        bias=bias, embedding=atac_embedding)
        hypers['pca_n_components'] = pca_n_components
        super().__init__(generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac, hypers=hypers)

class GanoliMLPGenerator(GanoliGAN):

    def __init__(self, input_shape, output_shape, input_modality='atac', hidden_dims=[], bias=True, **kwargs):
        super().__init__(input_modality=input_modality, **kwargs)
        self.leaky_relu = nn.LeakyReLU()
        layer_shapes = [input_shape, *hidden_dims, output_shape]
        self.layers = [nn.Linear(layer_shapes[i], layer_shapes[i+1], bias=bias) for i in range(len(hidden_dims+1))]

        def model(x):
            for layer in self.layers:
                x = layer(x)
                x = self.leaky_relu(x)
            return x # todo: think about tanh in generator

        self.model = model


if __name__ == '__main__':
    data_root = '/om2/user/rogerjin/data/Ben'
    data_path = opj(data_root, 'data_files_new.npz')
    data = np.load(data_path, allow_pickle=True)
    # gene_list = pd.read_csv(f'{data_root}/gene_list.csv', header=None)
    # chosen_genes = gene_list[data['rna_good_feats']]
    # gene_labels = list(chosen_genes[1])

#     data_dir='/om2/user/rogerjin/data/NeurIPS2021/phase2/predict_modality'
#     rna_dir=f'{data_dir}/openproblems_bmmc_multiome_phase2_rna'
#     atac_dir=f'{data_dir}/openproblems_bmmc_multiome_phase2_mod2'

#     rna = ad.read_h5ad(f'{rna_dir}/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod1.h5ad')
#     atac = ad.read_h5ad(f'{rna_dir}/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_train_mod2.h5ad')

#     train_rna, val_rna, train_atac, val_atac = train_test_split(rna, atac, test_size=0.2, random_state=42)
#     train_rna, val_rna, train_atac, val_atac = train_rna.X.toarray(), val_rna.X.toarray(), train_atac.X.toarray(), val_atac.X.toarray()

    kwargs = {}
    if torch.cuda.is_available():
        kwargs['gpus'] = -1

    # tb_logger = loggers.TensorBoardLogger("logs/linear")
    # tb_logger = loggers.TensorBoardLogger("logs/shallow/")
    # tb_logger = loggers.TensorBoardLogger("logs/linear_embed_corr/")
    # tb_logger = loggers.TensorBoardLogger("logs/logistic_lr=0.0002_beta1=0.5/")
    # tb_logger = loggers.TensorBoardLogger("logs/logistic_embed_corr/")
    # tb_logger = loggers.TensorBoardLogger("logs/logistic_embed_corr_lr=0.0002_beta1=0.5/")
    # tb_logger = loggers.TensorBoardLogger("logs/logistic_embed_pca_lr=0.0002_beta1=0.5/")
    # tb_logger = loggers.TensorBoardLogger("logs/logistic_embed_pca_corr_lr=0.0002_beta1=0.5/")

    # tb_logger = loggers.TensorBoardLogger("neurips/logistic_embed_pca_corr_lr=0.0002_beta1=0.5_ncomponents=20/")

    # tb_logger = loggers.TensorBoardLogger("logs/shallow_embed_corr/")
    # tb_logger = loggers.TensorBoardLogger("logs/shallow_embed_corr_lr=0.0002_beta1=0.5/")
    
    # tb_logger = loggers.TensorBoardLogger("logs/shallow_logistic_embed_pca_corr_lr=0.0002_beta1=0.5/")

    tb_logger = loggers.TensorBoardLogger("logs/debug/")

    checkpointer = ModelCheckpoint(monitor='checkpointer_objective',
            filename='step={step:02d}-epoch={epoch:02d}-val_oracle_rna={checkpointer_objective:.2f}',
                                   save_top_k=10, auto_insert_metric_name=False)

    trainer = Trainer(**kwargs, logger=tb_logger, callbacks=[checkpointer], check_val_every_n_epoch=3)

    train_rna = data['rna_train']
    train_atac = data['atac_train_small']
    val_rna = data['rna_test']
    val_atac = data['atac_test_small']

    def self_correlation(matrix, device='cuda:0'):
        matrix = torch.Tensor(matrix).to(device)
        # return matrix.T @ matrix
        return torch.corrcoef(matrix.T)

    def pca_embedding(matrix, transform, device='cuda:0'):
        return transform(matrix)

    def embedding(matrix):
        # return pca_embedding(matrix)
        return self_correlation(matrix)

    # rna_embedding = embedding(train_rna)
    # atac_embedding = embedding(train_atac)
    n_components = 20
    pca_rna = PCA(n_components=n_components)
    pca_atac = PCA(n_components=n_components)
    # pca_rna.fit_transform(train_rna)
    # pca_atac.fit_transform(train_atac)
    pca_rna.fit_transform(self_correlation(train_rna, device='cuda:0').detach().cpu().numpy())
    pca_atac.fit_transform(self_correlation(train_atac, device='cuda:0').detach().cpu().numpy())
    rna_embedding = torch.Tensor(pca_rna.components_.T).to('cuda:0')
    atac_embedding = torch.Tensor(pca_atac.components_.T).to('cuda:0')

    rna_dim = train_rna.shape[1]
    atac_dim = train_atac.shape[1]

    train_rna = GanoliUnimodalDataset(train_rna)
    train_atac = GanoliUnimodalDataset(train_atac)
    train_rna_atac = GanoliMultimodalDataset(rna=train_rna, atac=train_atac)

    val_rna = GanoliUnimodalDataset(val_rna)
    val_atac = GanoliUnimodalDataset(val_atac)
    val_rna_atac = GanoliMultimodalDataset(rna=val_rna, atac=val_atac)

    batch_size = 32
    train_dataloader = DataLoader(train_rna_atac, batch_size=batch_size, num_workers=4)
    val_dataloader = DataLoader(val_rna_atac, batch_size=batch_size, num_workers=4)


    # gan = GanoliLinearGAN(7445, 3808)
    # gan = GanoliLogisticGAN(7445, 3808)
    # gan = GanoliShallowGAN(7445, 3808)
    # gan = GanoliLinearGAN(7445, 3808, rna_embedding=rna_embedding, atac_embedding=atac_embedding)
    # gan = GanoliLogisticGAN(7445, 3808, rna_embedding=rna_embedding, atac_embedding=atac_embedding)
    # gan = GanoliShallowGAN(7445, 3808, rna_embedding=rna_embedding, atac_embedding=atac_embedding, rna_embedding_labels=gene_labels)

    gan = GanoliShallowLogisticPCAGAN(rna_dim, atac_dim, rna_embedding=rna_embedding, atac_embedding=atac_embedding, pca_n_components=n_components)

    trainer.fit(gan, train_dataloader, val_dataloader)
