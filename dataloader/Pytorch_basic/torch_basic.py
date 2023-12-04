import torch
from torch.nn.utils.rnn import pad_sequence

# loader
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, data, 
                 shuffle = True, batch_size = 256, num_workers = 20, 
                 Dataset = None, Collate_fn = None, 
                 dataset_kwargs = {}, collate_fn_kwargs = {}, 
                 *args, **kwargs):
        self.dataset = Dataset(data, **dataset_kwargs)
        if Collate_fn is not None:
            Collate_fn = Collate_fn(**collate_fn_kwargs)
        super(DataLoader, self).__init__(
            self.dataset,
            collate_fn = Collate_fn,
            shuffle = shuffle,
            batch_size = batch_size,
            num_workers = num_workers,
            *args, **kwargs)


# dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self):


    def __getitem__(self, i):

        return 
    
    def __call__(self):
        pass
    
    def __len__(self):
        return len()

# collate function
class Collate_fn:
    def __init__(self, pad_value):
        self.pad_value = pad_value
    
    def __call__(self, batch):
        x_batch, y_batch = [], []
        batch_max_length = 0
        for x, y in batch:
            x_batch.append(x)
            y_batch.append(y)
            #batch_max_length = batch_max_length if batch_max_length > length else length
        
        #x_batch = torch.stack(x_batch)
        #x_batch = pad_sequence(x_batch, padding_value = self.pad_value, batch_first = True)
        #y_batch = torch.stack(y_batch)
        return x_batch, y_batch, batch_max_length 