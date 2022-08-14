import torch

def add_pad_embedding(embeddings):
    return torch.cat([embeddings, torch.zeros(1, embeddings.shape[1])])


def pad(vec, seq_len, pad_token=-1):
    '''
    1. turn nonzeros into indices
    2. use indices as in normal transformer to get embedding,
    pretty much like token indices - should I include padding tokens?
    3. but also somehow use these indices to extract the output
    4. it may be that I can also accomplish this via masking
    '''
    return torch.cat([
        vec, torch.Tensor([pad_token]).expand(seq_len - vec.size()[0])
    ])

def select_nonzero(seq, pad_token=-1):
    seqs = []
    for row in seq:
        nonzeros = pad(row.nonzero().T[0], seq.shape[-1], pad_token)
        seqs.append(nonzeros)
    return torch.stack(seqs)

def num_embeddings(embeddings):
    return embeddings.weight.size()[0] - 1

def squish_and_embed(seq, embeddings):
    nonzeros = select_nonzero(seq, pad_token=num_embeddings(embeddings))
    return embeddings(nonzeros.long()) * nonzeros.unsqueeze(-1)

if __name__ == '__main__':
    seq = torch.Tensor([
    [0, 1, 0, 10, 100],
    [1, 0, 0, 10, 0]
])

    embeddings = torch.Tensor([
        [0, 1, 1,],
        [2, 3, 3],
        [4, 5, 5],
        [6, 7, 7],
        [8 ,9, 9],
    ])

    embeddings = add_pad_embedding(embeddings)
    embeddings = torch.nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=-1)
    print(embeddings)
    print(select_nonzero(seq))
    print(squish_and_embed(seq, embeddings))
