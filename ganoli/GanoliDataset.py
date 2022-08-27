from torch.utils.data import Dataset

class GanoliUnimodalDataset(Dataset):
    
    def __init__(self, matrix):
        self.matrix = matrix
        
    def __len__(self):
        return self.matrix.shape[0]
    
    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError
            
        return self.matrix[idx]

class GanoliMultimodalDataset(Dataset):

    def __init__(self, **datasets):
        self.datasets = datasets
        self._len = min(len(d) for d in self.datasets.values())

    def __getitem__(self, i):
        # return tuple(d[i] for d in self.datasets)
        return {name: dataset[i] for name, dataset in self.datasets.items()}

    def __len__(self):
        return self._len
