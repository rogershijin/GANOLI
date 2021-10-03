import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import MSELoss

class GanoliModel(pl.LightningModule):
    
    def __init__(self, generator_rna2atac, generator_atac2rna, discriminator_rna):
        super().__init__()
        self.generator_rna2atac = generator_rna2atac
        self.generator_atac2rna = generator_atac2rna
        self.discriminator_rna = discriminator_rna
        self.discriminator_atac = discriminator_atac
    
    def forward(self, rna_or_atac, data_type):
        if data_type == 'rna':
            return generator_rna2atac(rna_or_atac)
        
        if data_type == 'atac':
            return generator_atac2rna(rna_or_atac)
        
        else:
            raise ValueError(f"Invalid datatype {datatype}, expected 'rna' or 'atac'")
            
    def training_step(self, batch, batch_idx, optimizer_idx):
        rna_real, atac_real = batch
        
        rna_generated = generator_atac2rna(atac_real)
        atac_generated = generator_rna2atac(rna_real)
        
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
    
    def __init__(self, input_modality='atac', input_shape, output_shape, bias=True):
        super().__init__(input_modality=input_modality)
        
        self.model = nn.Linear(input_shape, output_shape, bias=bias)
    
    
class GanoliLinearDiscriminator(GanoliDiscriminator):
    
    def __init__(self, input_modality='atac', input_shape, bias=True):
        super().__init__(input_modality=input_modality)
        
        self.model = nn.Linear(input_shape, 1, bias=bias)
        
        
class GanoliLinearGAN(GanoliLinearGAN):
    
    def __init__(self, rna_shape, atac_shape, bias=True):
        pass