import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from SquishTransformer import SquishTransformer
from MLP import MLP
import numpy as np
import scanpy as sc
from muon import MuData
import wandb
import pprint
import os
from squish_indexing import squish_and_embed
from torch.nn import MSELoss
from torch.optim import AdamW
import math
import argparse
import json
from time import perf_counter

# end any ongoing wandb runs
try:
    wandb.finish()
except:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()
config_path = args.config
config = json.load(open(config_path))

wandb.init(project="Squish Transformer", entity="rogershijin", config=config, name=config.get('run_name', None))
checkpoint_base_dir = config.get('checkpoint_base_dir', '/om2/user/rogerjin/checkpoints')
checkpoint_dir = f'{checkpoint_base_dir}/{wandb.run.name}'
os.makedirs(checkpoint_dir, exist_ok=True)
wandb.config.update({'checkpoint_dir': checkpoint_dir})
pprint.pprint(dict(wandb.config), indent=2)

remote_data_dir = config.get('data_dir', '/om2/user/rogerjin/data/NeurIPS2021/multiome')
remote_atac_dir = f'{remote_data_dir}/atac'
remote_rna_dir = f'{remote_data_dir}/rna'
remote_atac_path = f'{remote_data_dir}/multiome_atac_processed_training.h5ad'
remote_rna_path = f'{remote_data_dir}/multiome_gex_processed_training.h5ad'
cache_dir="/om2/user/rogerjin/.cache"



# atac = {
#     'train': sc.read_h5ad(f'{remote_atac_dir}/atac_train_sorted_decreasing_variance.h5ad'),
#     'val': sc.read_h5ad(f'{remote_atac_dir}/atac_val_sorted_decreasing_variance.h5ad'),
#     'test': sc.read_h5ad(f'{remote_atac_dir}/atac_test_sorted_decreasing_variance.h5ad')
# }

atac = {
    'train': sc.read_h5ad(f'{remote_atac_dir}/atac_train.h5ad'),
    'val': sc.read_h5ad(f'{remote_atac_dir}/atac_val.h5ad'),
    'test': sc.read_h5ad(f'{remote_atac_dir}/atac_test.h5ad')
}

rna = {
    'train': sc.read_h5ad(f'{remote_rna_dir}/rna_train.h5ad'),
    'val': sc.read_h5ad(f'{remote_rna_dir}/rna_val.h5ad'),
    'test': sc.read_h5ad(f'{remote_rna_dir}/rna_test.h5ad')
}

MODELS = {
    'squish_transformer': SquishTransformer,
    'mlp': MLP,
}

class MuDataWithLen(MuData):
    
    def __len__(self):
        return self.n_obs

datasets = {
    partition: MuDataWithLen({'atac': atac[partition], 'rna': rna[partition]}) for partition in atac.keys()
}

torch.manual_seed(42)

samplers = {
    'train': RandomSampler,
    'val': SequentialSampler,
    'test': SequentialSampler
}

def collate_fn(x):
    return x[0]

# todo: increase val/test batch size

# todo: debug num_workers>0 problem
loaders = {
    partition: DataLoader(dataset, sampler=BatchSampler(samplers[partition](dataset), batch_size=config['batch_size'], drop_last=False), num_workers=0, collate_fn=collate_fn) for partition, dataset in datasets.items()
}

model_name = config.get('model', 'squish_transformer').lower()
model_class = MODELS[model_name]
model_kwargs = config.get('model_kwargs', {})
model = model_class(**model_kwargs)
# device = 'cpu'
device = 'cuda:0'
model.to(device)

loss_fn = MSELoss()
optimizer = AdamW(model.parameters(), lr=config['lr'])

def forward_pass(batch, use_binary=config.get('use_binary', False)):
    if use_binary:
        atac = batch.mod['atac'].X.tocsr().tocoo()
    else:
        atac = batch.mod['atac'].layers['counts'].tocsr().tocoo()
    if model_name == 'mlp':
        atac = torch.tensor(atac.todense(), dtype=torch.float).to(device)
        return model(atac)
    if model_kwargs.get('transformer', '') == 'google/long-t5-local-base':
        model_embeddings = model.transformer.embed_tokens
    else:
        model_embeddings = model.transformer.embeddings.word_embeddings
    squished = squish_and_embed(atac, model_embeddings, max_seq_len=config['max_seq_len'])
    out = model(inputs_embeds=squished['embeddings'], attention_mask=squished['attention_mask'])
    return out
    
def train_step(batch):
    optimizer.zero_grad()
    out = forward_pass(batch)
    rna = torch.tensor(batch.mod['rna'].X.todense()).float().to(device)
    loss = loss_fn(out, rna)
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_batch(batch):
    with torch.no_grad():
        out = forward_pass(batch)
        rna = torch.tensor(batch.mod['rna'].X.todense()).float().to(device)
        loss = loss_fn(out, rna)
    return loss.item()

best_checkpoint_path = ''

best_train_loss = math.inf
best_train_loss_epoch = 0

print(f'STARTING INITIAL VALIDATION...')
val_loss = 0
val_start = perf_counter()
for batch in loaders['val']:
    val_loss += batch.n_obs * eval_batch(batch) / len(datasets['val'])
    break
val_end = perf_counter()
print(f'val_loss={val_loss:.5f} time={(val_end - val_start)/60:.5f}')
print(f'INITIAL VALIDATION COMPLETE\n')
    
best_val_loss = val_loss
best_val_loss_epoch = 0
    
wandb.log({
    'epoch': 0,
    'val/loss': val_loss,
    'val/best_loss': best_val_loss,
    'val/time': (val_end - val_start)/60
})

print(f'STARTING TRAINING...')
for epoch in range(1, config['epochs']+1):
    train_loss = 0
    val_loss = 0
    
    train_start = perf_counter()
    for batch in loaders['train']:
        train_loss += batch.n_obs * train_step(batch) / len(datasets['train'])
    train_end = perf_counter()
        
    val_start = perf_counter()
    for batch in loaders['val']:
        val_loss += batch.n_obs * eval_batch(batch) / len(datasets['val'])
    val_end = perf_counter()
        
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        best_train_loss_epoch = epoch
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_loss_epoch = epoch
        print(f'Epoch {epoch}/{config["epochs"]}: New best val loss of {val_loss:.5f}!')
        
        if epoch > 5:
            best_checkpoint_path = f"{wandb.config.checkpoint_dir}/epoch={epoch}-val_loss={val_loss:.5f}.pt" 
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train/loss': train_loss,
                'train/best_loss': best_train_loss,
                'train/best_loss_epoch': best_train_loss_epoch,
                'val/loss': val_loss,
                'val/best_loss': best_val_loss,
                'val/best_loss_epoch': best_val_loss_epoch,
                }, best_checkpoint_path)

            with open(f"{wandb.config.checkpoint_dir}/best_checkpoint_path.txt", 'w') as best_f:
                best_f.write(best_checkpoint_path)
            
            print(f'Saved checkpoint to: {best_checkpoint_path}')
        
    wandb.log({
        'epoch': epoch,
        'train/loss': train_loss, 
        'train/best_loss': best_train_loss,
        'train/best_loss_epoch': best_train_loss_epoch,
        'train/time': (train_end - train_start)/60,
        'val/loss': val_loss,
        'val/best_loss': best_val_loss,
        'val/best_loss_epoch': best_val_loss_epoch,
        'val/time': (val_end - val_start)/60,
        'best_checkpoint_path': best_checkpoint_path,
    })
    
    print(f'Epoch {epoch}/{config["epochs"]}: train_loss={train_loss:.5f} best_train_loss={best_train_loss:.5f} best_train_loss_epoch={best_train_loss_epoch} val_loss={val_loss:.5f} best_val_loss={best_val_loss:.5f} best_val_loss_epoch={best_val_loss_epoch} time={(val_end-train_start)/60:.5f}')
    
test_loss = 0
test_start = perf_counter()
for batch in loaders['test']:
    test_loss += batch.n_obs * eval_batch(batch) / len(datasets['test'])
test_end = perf_counter()
    
wandb.log({
    'epoch': epoch,
    'test/loss': test_loss,
    'test/time': (test_end-test_start)/60
})
print('TRAINING COMPLETE\n')

print('STARTING TESTING...')
print(f'test_loss={test_loss:.5f} time={(test_end-test_start)/60:.5f}')
print('TESTING COMPLETE')

wandb.finish()
