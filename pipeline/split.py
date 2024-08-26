from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

def split_data(df, column_of_interest, tokenizer, cross_dressed_df):
    print("CROSS DRESSED SHAPE")
    print(cross_dressed_df.shape)
    # Split data into training and dev and test
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=0)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=0)

    #double check if the cross dressed data is being added to the validation set
    print("BEFORE ADDING CROSS DRESSED")
    print(val_data.shape)
    val_data = pd.concat ([cross_dressed_df,val_data])
    
    print("AFTER ADDING CROSS DRESSED")
    print(val_data.shape)

    print(train_data.shape)

    train_encodings = tokenizer(list(train_data[column_of_interest]), truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(list(test_data[column_of_interest]), truncation=True, padding=True, return_tensors='pt')
    val_encodings = tokenizer(list(val_data[column_of_interest]), truncation=True, padding=True, return_tensors='pt')
    

    # Create PyTorch datasets
    class CustomDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_dataset = CustomDataset(train_encodings, list(train_data['is_male']))
    test_dataset = CustomDataset(test_encodings, list(test_data['is_male']))
    val_dataset = CustomDataset(val_encodings, list(val_data['is_male']))

    print(len(train_dataset), len(test_dataset), len(val_dataset))
    return train_dataset, test_dataset, train_data, test_data, val_dataset, val_data