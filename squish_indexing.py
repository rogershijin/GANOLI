import torch
import functorch
import scipy.sparse

DEFAULT_ARANGE_LEN = 10
COUNT_PAD_TOKEN = 0
BATCH_SIZE = 8


def add_pad_embedding(embeddings):
    return torch.cat([embeddings, torch.zeros(1, embeddings.shape[1])])


def pad(vec, seq_len, pad_token=-1):
    """
                    1. turn nonzeros into indices
                    2. use indices as in normal transformer to get embedding,
    pretty much like token indices - should I include padding tokens?
                    3. but also somehow use these indices to extract the output
                    4. it may be that I can also accomplish this via masking
    """
    return torch.cat([vec, torch.tensor([pad_token]).expand(seq_len - vec.size()[0])])


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
    return seq.T[indices.long()].permute(0, 2, 1).diagonal().T


# def cyclic_arange(lengths, arange=torch.arange(DEFAULT_ARANGE_LEN)):
#     max_len = lengths.max().item()
#     if max_len > DEFAULT_ARANGE_LEN:
#         arange = torch.arange(max_len)
#     arange_vmap = functorch.vmap(lambda length: (arange + 1) * (arange < length))
#     vmapped = arange_vmap(lengths).flatten()
#     return vmapped[vmapped.nonzero()].flatten() - 1


def cyclic_arange(lengths):
    if lengths.dim() == 1:
        lengths = lengths.unsqueeze(-1)
    max_len = lengths.max().item()
    arange = torch.arange(max_len).unsqueeze(0)
    padded_aranges = ((arange + 1) * (arange < lengths)).flatten()
    return padded_aranges[padded_aranges.nonzero()].flatten() - 1


def index_and_pad(indices, data, pad_token):
    return torch.sparse_coo_tensor(indices, data - pad_token).float().to_dense() + pad_token


def _squish_and_embed_sparse(
    batch_coo, embedding, batch_size=BATCH_SIZE, index_pad_token="last"
):
    '''
    batch_coo: sparse matrix in coo format. batch_coo.row must be sorted. this can be accomplished by calling batch_coo.to_csr().to_csc()
    '''
    if index_pad_token == "last":
        index_pad_token = embedding.num_embeddings - 1
    rows = torch.tensor(batch_coo.row)
    _, num_nonzeros = torch.unique_consecutive(rows, return_counts=True)
    assert _.shape[0] == batch_size  # otherwise there's an all-0 sequence in the batch
    sparse_indices = torch.stack([rows, cyclic_arange(num_nonzeros)])
    squish_indices = index_and_pad(
        sparse_indices, batch_coo.col, index_pad_token
    ).long()
    counts = index_and_pad(sparse_indices, batch_coo.data, COUNT_PAD_TOKEN)
    squish_indices = squish_indices.to('cuda:0')
    counts = counts.to('cuda:0')
    embeddings = embedding(squish_indices) * counts.unsqueeze(-1)
    return {
        "indices": squish_indices,
        "counts": counts,
        "attention_mask": counts > 0,
        "embeddings": embeddings
    }


def squish_and_embed(batch, embeddings, batch_size=BATCH_SIZE):
    if scipy.sparse.issparse(batch):
        return _squish_and_embed_sparse(batch, embeddings, batch_size=batch_size)
    nonzero_indices, nonzero_seq = select_nonzero(
        batch, pad_token=num_embeddings(embeddings)
    )
    return {
        "indices": nonzero_indices,
        "counts": nonzero_seq,
        "attention_mask": counts > 0,
        "embeddings": embeddings(nonzero_indices.long()) * nonzero_seq.unsqueeze(-1),
    }


if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with record_function("all"):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"using device '{device}'")

            seq = torch.tensor([[0, 1, 0, 10, 100], [1, 0, 0, 10, 0]])
            sparse_seq = scipy.sparse.coo_matrix(seq)

            embeddings = torch.tensor(
                [
                    [0, 1, 1],
                    [2, 3, 3],
                    [4, 5, 5],
                    [6, 7, 7],
                    [8, 9, 9],
                ]
            )

            seq.to(device)
            embeddings.to(device)

            embeddings = add_pad_embedding(embeddings)
            embeddings = torch.nn.Embedding.from_pretrained(
                embeddings, freeze=False, padding_idx=-1
            )

            nonzero_ans = torch.tensor([[1, 3, 4], [0, 3, 5]])  # 5 is padding token

            squish_and_embed_ans = torch.tensor(
                [
                    [[2.0, 3.0, 3.0], [60.0, 70.0, 70.0], [800.0, 900.0, 900.0]],
                    [[0.0, 1.0, 1.0], [60.0, 70.0, 70.0], [0.0, 0.0, 0.0]],
                ]
            )
            try:
                out = squish_and_embed(seq, embeddings)
                assert torch.equal(out["indices"], nonzero_ans)
                assert torch.equal(out["embeddings"], squish_and_embed_ans)
                print("passed dense cases")
            except Exception as e:
                print("failed at least one dense case")
            try:
                out = squish_and_embed(sparse_seq, embeddings, batch_size=2)
                assert torch.equal(out["indices"], nonzero_ans)
                assert torch.equal(out["embeddings"], squish_and_embed_ans)
                print("passed sparse cases")
            except Exception as e:
                print("failed at least one sparse case")
                raise e

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))