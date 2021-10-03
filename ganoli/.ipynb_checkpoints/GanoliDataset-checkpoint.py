from torch.utils.data import Dataset

class GanoliDataset(Dataset):
    
    def __init__(self, matrix):
        self.matrix = matrix
        
    def __len__(self):
        return self.matrix.shape[0]
    
    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError
            
        return self.matrix[idx]
        