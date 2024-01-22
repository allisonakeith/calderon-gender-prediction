# %% [markdown]
# I'll start by importing the necessary libraries for text preprossessing

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# Now I'll import the libraries for the machine learning part

# %%
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# Now I'm going to import the dataset and make sure I'm only using the appropriate rows (only comedias, and only characters that have some dialogue)

# %%
character_file = '/Users/allisonkeith/calderon-gender-prediction/calderon-gender-prediction/character_utterances.csv'

character_df = pd.read_csv(character_file, usecols = ['id','character_gender','character_id', 'character_sentences', 'character_utterances', 'genre', 'tokens', 'tokens_length'])

#examine only the comedias

comedias_df = character_df[(character_df.genre != 'auto sacramental') & (character_df.genre != 'loa') & (character_df.genre != 'auto sacramental - loa') & (character_df.character_gender != 'UNKNOWN')]

#drop nan values
comedias_df = comedias_df.dropna()
comedias_df = comedias_df[comedias_df['tokens_length'] > 10]
print(comedias_df.shape)

# %% [markdown]
# Now I'm gonna make a column called is_male, which is the column of interest for the classifier

# %%
#########################
#create a column called "is_male"
#########################
comedias_df['is_male'] = np.where(comedias_df['character_gender'] == 'MALE', 1, 0)
print(comedias_df.head())

# %% [markdown]
# ## Creating the Test and Training Sets

# %% [markdown]
# Fist I want to ensure that the cross_dressed characters are removed from the training set, because we want to test on them later

# %%
cross_dressed_characters = ['lindabridis', 'claridiana', 'rosaura', 'eugenia', 'semíramis', 'ninias']
cross_dressed_df = comedias_df[comedias_df['character_id'].isin(cross_dressed_characters)]
print(cross_dressed_df.shape)

tokens_df = comedias_df[~comedias_df['character_id'].isin(cross_dressed_characters)]
print(tokens_df.shape)

######################################
#set the model path
######################################
model_path = "citizenlab/distilbert-base-multilingual-cased-toxicity"

############################
#Split the data into training and testing sets using sklearn
############################



def split_data(df, column_of_interest):
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=0)
    test_data = pd.concat ([test_data, cross_dressed_df])

    print(train_data.shape)


    # Load tokenizer and encode the text
    #tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll02-spanish")
    #tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    train_encodings = tokenizer(list(train_data[column_of_interest]), truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(list(test_data[column_of_interest]), truncation=True, padding=True, return_tensors='pt')
    

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
    return train_dataset, test_dataset, train_data, test_data

# %% [markdown]
# Now I'm going to create the training and test sets, based on the data set that we want to test

# %%
# Assuming 'gender' is the target category and 'tokens' is the text
df = comedias_df[['id','character_id','is_male', 'tokens']]


cross_dressed_df = cross_dressed_df[['id','character_id','is_male', 'tokens']]

column_of_interest = 'tokens'

train_dataset, test_dataset, train_data, test_data = split_data(df, column_of_interest)

# %% [markdown]
# ## Set the Model Parameters

# %% [markdown]
# #https://www.youtube.com/watch?v=c36lUUr864M
# #https://huggingface.co/docs/transformers/main_classes/pipelines 
# 
# #https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish
# #or
# #https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli
# 

# %%


# Load pre-trained model
#model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large-finetuned-conll02-spanish", num_labels=2)
#model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=2)   
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define optimizer, loss is already specified as crossEntropyLoss in the model
optimizer = Adam(model.parameters(), lr=1e-5)

batch_size = 16
num_epochs = 10

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)

# %%
#######################
# Define a Function to Test the model
#######################
def test_model(model, test_dataset):
    # Evaluation loop
    
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        model.eval()
        all_predictions = []
        all_probabilities = []
        for batch in DataLoader(test_dataset, batch_size=16):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)


            probabilities = torch.nn.functional.softmax(logits, dim=1)

            print("Logits:", logits)
            print("Probabilities:", probabilities)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    accuracy = total_correct / total_samples
    return accuracy, all_predictions, all_probabilities

# # %%
# accuracy, all_predictions = test_model(model, test_dataset)
# print("Accuracy: ", accuracy)

# #write predictions into the dataframe
# test_data['predictions'] = all_predictions

# #test_data.to_csv('/projekte/tcl/users/keithan/projectcalderon/wp1-semantic-analysis/character_analysis/gender_analysis/gender_classifier/results/before_training_gender_predictions.csv', index=False)
# print(test_data['predictions'].value_counts())

# %%
def train_model(model, train_dataset, test_dataset):
    for epoch in range(10):
        model.train()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for batch in DataLoader(train_dataset, batch_size=16, shuffle=True):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

        # Calculate epoch statistics
        epoch_accuracy = total_correct / total_samples
        epoch_loss = total_loss / len(train_dataset)

        # Print epoch statistics
        print(f'Epoch {epoch + 1}/{5}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2%}')
        
        print("True labels:", labels)
        print("Predictions:", predictions)  

    return model

# %%
trained_model = train_model(model, train_dataset, test_dataset)

# %%
accuracy, all_predictions, all_probabilities = test_model(trained_model, test_dataset)
print("Accuracy: ", accuracy)

# %%
#write predictions into the dataframe
test_data['predictions'] = all_predictions
test_data['probabilities'] = all_probabilities

test_data.to_csv('/projekte/tcl/users/keithan/projectcalderon/wp1-semantic-analysis/character_analysis/gender_analysis/gender_classifier/results/tokens_distilbert_stratified_gender_predictions.csv', index=False)

print(test_data['predictions'].value_counts())


