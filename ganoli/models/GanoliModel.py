import sys
sys.path.append('/om2/user/rogerjin/GANOLI/ganoli')
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from os.path import join as opj
import numpy as np
from GanoliDataset import GanoliDataset


seed_everything(42, workers=True)


class GanoliGAN(pl.LightningModule):
    
    def __init__(self, generator_rna2atac, generator_atac2rna, discriminator_rna, discriminator_atac):
        super().__init__()
        self.generator_rna2atac = generator_rna2atac
        self.generator_atac2rna = generator_atac2rna
        self.discriminator_rna = discriminator_rna
        self.discriminator_atac = discriminator_atac
    
    def forward(self, rna_or_atac, data_type):
        if data_type == 'rna':
            return self.generator_rna2atac(rna_or_atac)
        
        if data_type == 'atac':
            return self.generator_atac2rna(rna_or_atac)
        
        else:
            raise ValueError(f"Invalid datatype {data_type}, expected 'rna' or 'atac'")
            
    def training_step(self, batch, batch_idx, optimizer_idx):
        rna_real, atac_real = batch
        
        rna_generated = self.generator_atac2rna(atac_real)
        atac_generated = self.generator_rna2atac(rna_real)
        
        mse = MSELoss()
        loss = mse(rna_generated, rna_real) + mse(atac_generated + atac_real)
        
        return loss
    
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
    train_rna = GanoliDataset(data['rna_train'])
    train_atac = GanoliDataset(data['atac_train_small'])
#     test_dataset = GanoliDataset(rna_test)
    train_rna__loader = DataLoader(train_rna)
    train_atac_loader = DataLoader(train_atac)
#     test_loader = DataLoader(test_dataset)
    trainer.fit(gan, train_dataloader)
