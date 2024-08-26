from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ast


from split import split_data
from test_train import test_model, train_model
from interpret import interpret_model


##################################################

comedias_df = pd.read_csv('calderon-gender-predict/masked_comedias.csv')

##################################################
cross_dressed_characters = ['lindabridis', 'claridiana', 'rosaura', 'eugenia', 'semíramis', 'ninias']
cross_dressed_df = comedias_df[comedias_df['character_id'].isin(cross_dressed_characters)]
print(cross_dressed_df.shape)

tokens_df = comedias_df[~comedias_df['character_id'].isin(cross_dressed_characters)]
print(tokens_df.shape)

##################################################


#first, split up the utterance and scene data
def separate_list(comedias_df, column_of_interest):

    column_data = []
    for index, row in comedias_df.iterrows():
        is_male = row['is_male']
        id = row['id']
        character_id = row['character_id']
        character_speech_str = row[column_of_interest]
        character_speech_list = ast.literal_eval(character_speech_str)

        #set min length of speech to 10 words
        character_speech_list = [speech for speech in character_speech_list if len(speech) > 30]

        for speech in character_speech_list:
            speech = ''.join(speech)
            column_data.append({'is_male': is_male, column_of_interest: speech, 'id': id, 'character_id': character_id})
            #only use column data where speech is longer than 10 words


    tokens_df = pd.DataFrame(column_data)
    return tokens_df














##################################################
model_path =  'dccuchile/bert-base-spanish-wwm-cased' #'distilbert-base-multilingual-cased'
model_name = 'bert-base-spanish' #'distilbert-multilingual'

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


num_epochs = 12
lr = 0.00001
batch_size = 32 #24

# Define optimizer, loss is already specified as crossEntropyLoss in the model


class_weights = torch.tensor([1.0,2.0]).to(device) #this doesn't get used in the model

for column_of_interest in ['tokens','scenes', 'utterances']: #'tokens',,'utterances''tokens'
    if column_of_interest == 'tokens':
        learning_rate = 0.00001
        batch_size = 24
        num_epochs = 12
    if column_of_interest == 'scenes':
        learning_rate = 0.00001
        batch_size = 32
        num_epochs = 12
    if column_of_interest == 'utterances':
        learning_rate = 0.00001
        batch_size = 32
        num_epochs = 19
    print(f'Column of interest: {column_of_interest}')
    optimizer = Adam(model.parameters(), lr=lr)
    model.to(device)

    if column_of_interest == 'utterances' or column_of_interest == 'scenes':
        tokens_df = separate_list(tokens_df, column_of_interest)
        cross_dressed_df = separate_list(cross_dressed_df, column_of_interest)

    


    train_dataset, test_dataset, train_data, test_data, val_dataset, val_data = split_data(tokens_df, column_of_interest, tokenizer, cross_dressed_df)
    trained_model, best_epoch = train_model(model, train_dataset, val_dataset, num_epochs, batch_size, optimizer, device, pos_weight=class_weights)

    accuracy, all_predictions, all_probabilities = test_model(trained_model, val_dataset, batch_size, device)
                            
    val_data['predictions'] = all_predictions
    val_data['probabilities'] = all_probabilities


    file_name = f"calderon-gender-prediction/results/june7_{column_of_interest}_{model_name}_{lr}_{batch_size}_{best_epoch}.csv"
    val_data.to_csv(file_name, index=False)

    attribution = 1


    #if statement in case I want to run one column at a time
    if column_of_interest == 'tokens' or column_of_interest == 'scenes' or column_of_interest == 'utterances':
        
        #create a list to store word attributions
        word_attributions = []
        text_list = []
        #interpret the model
        for index, row in val_data.iterrows():
            #### 
            text_list.append(row[column_of_interest])

            ####
            #only take the first 512 tokens of the speech
            word_attribution = interpret_model(trained_model, text_list, tokenizer, visualizer = False)
            word_attributions.append(word_attribution)

        #output word attributions to a file
        word_attributions_df = pd.DataFrame(word_attributions)
        word_attributions_df.to_csv(f"calderon-gender-prediction/results/june7_{column_of_interest}_word_attributions.csv", index=False)
        attribution += 1

