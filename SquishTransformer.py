import torch
from transformers import DistilBertModel, LongT5Model
from torch.nn import Linear, ReLU

transformers = {
    'distilbert-base-uncased': DistilBertModel,
    'google/long-t5-local-base': LongT5Model
}

class Zero(torch.nn.Module):
    
    def forward(self, batch):
        return 0

class SquishTransformer(torch.nn.Module):
    
    def __init__(self, output_dim=13431, transformer='distilbert-base-uncased', cache_dir=None): # todo: cache dir
        super().__init__()
        self.output_dim = output_dim
        self.relu = ReLU()
        if transformer == 'distilbert-base-uncased':
            self.transformer = DistilBertModel.from_pretrained(transformer, cache_dir=cache_dir)
            self.transformer.embeddings.word_embeddings = torch.nn.Embedding(116492, 768) # todo: magic numbers
            self.transformer.embeddings.position_embeddings = Zero()
            self.pre_classifier = Linear(self.transformer.config.dim, self.transformer.config.dim)
            self.classifier = Linear(self.transformer.config.dim, output_dim)
        elif transformer == 'google/long-t5-local-base':
            self.transformer = LongT5Model.from_pretrained("google/long-t5-local-base").encoder
            self.transformer.embed_tokens = torch.nn.Embedding(116492, 768)
            self.pre_classifier = Linear(768, 768)
            self.classifier = Linear(768, output_dim)
            
        
    def forward(self, **kwargs):
        out = self.transformer(**kwargs).last_hidden_state[:, 0] # embedding of cls
        out = self.pre_classifier(out)
        out = self.relu(out)
        out = self.classifier(out)
        out = self.relu(out)
        return out