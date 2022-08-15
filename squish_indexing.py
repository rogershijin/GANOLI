import torch
import torch.jit as jit

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
        vec, torch.tensor([pad_token]).expand(seq_len - vec.size()[0])
    ])

def select_nonzero(seq, pad_token=-1):
    if pad_token < 0:
        # +1 because we add a 0 row to embedding matrix
        pad_token = pad_token % seq.shape[-1] + 1
    unpadded_indices = []
    indices = []
    seqs = []
    seq_len = 0
    for row in seq:
        nonzero_indices = row.nonzero(as_tuple=False).T[0]
        unpadded_indices.append(nonzero_indices)
        seq_len = max(seq_len, nonzero_indices.shape[-1])
    for nonzero_indices, row in zip(unpadded_indices, seq):
        nonzero_elts = pad(row[nonzero_indices], seq_len, 0)
        nonzero_indices = pad(nonzero_indices, seq_len, pad_token)
        indices.append(nonzero_indices)
        seqs.append(nonzero_elts)
    return torch.stack(indices), torch.stack(seqs)

def num_embeddings(embeddings):
    return embeddings.weight.size()[0] - 1

def index_tensor(seq, indices):
    return seq.T[indices.long()].permute(0,2,1).diagonal().T

def squish_and_embed(seq, embeddings):
    nonzero_indices, nonzero_seq = select_nonzero(seq, pad_token=num_embeddings(embeddings))
    return embeddings(nonzero_indices.long()) * nonzero_seq.unsqueeze(-1)

if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("all"):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"using device '{device}'")

            seq = torch.tensor([
            [0, 1, 0, 10, 100],
            [1, 0, 0, 10, 0]
        ])

            embeddings = torch.tensor([
                [0, 1, 1,],
                [2, 3, 3],
                [4, 5, 5],
                [6, 7, 7],
                [8 ,9, 9],
            ])

            seq.to(device)
            embeddings.to(device)

            embeddings = add_pad_embedding(embeddings)
            embeddings = torch.nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=-1)

            nonzero_ans = torch.tensor([[ 1,  3,  4],
                                        [ 0,  3, 5]])
            squish_and_embed_ans = torch.tensor( [[[  0.,   1.,   1.],
                                             [ 20.,  30.,  30.],
                                             [400., 500., 500.]],

                                            [[  0.,   1.,   1.],
                                             [ 20.,  30.,  30.],
                                             [  0.,   0.,   0.]]])
            try:
                indices, seq = select_nonzero(seq)
                assert torch.equal(indices, nonzero_ans)
            except:
                print("error in nonzero selection")
                print('produced')
                print(indices)
                print('expected')
                print(nonzero_ans)
            try:
                assert torch.equal(squish_and_embed(seq, embeddings), squish_and_embed_ans)
            except:
                print("error in indexing")
                print('produced')
                print(squish_and_embed(seq, embeddings))
                print('expected')
                print(squish_and_embed_ans)


    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
