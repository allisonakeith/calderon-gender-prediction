import pandas as pd
import os
import spacy
import xml.etree.ElementTree as ET

input_directory = '/Users/allisonkeith/calderon-gender-prediction/calderon-gender-prediction/tei_plays'

output_CSV = '/Users/allisonkeith/calderon-gender-prediction/calderon-gender-prediction/gender_prediction_data.csv'

# function that lemmatizes a string of text
nlp = spacy.load('es_core_news_sm')

def lemmatize(text):
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.pos_ == 'SPACE' or token.pos_ == 'PUNCT':
            continue
        else:
            lemmas.append(token.lemma_)
    return lemmas

# Function that produces a list of dictionaries with character information from the xml file
def get_speaker_tokens(root):
    characterlist = []
    ns = {'tei': 'http://www.tei-c.org/ns/1.0', 'xml': 'http://www.w3.org/XML/1998/namespace'}

    for person in root.findall('.//tei:person', ns):
        name = person.find('.//tei:persName', ns).text
        xml_id = person.get('{http://www.w3.org/XML/1998/namespace}id')
        if xml_id is not None:
            characterdict = {'id': xml_id, 'Name': name, 'gender': person.get('sex'), 'tokens_length': '', 'tokens': [], 'lemmas': []}
            tokens = []
            lemmas = []
            for speech in root.findall('.//tei:sp', ns):
                speech_who = speech.get('who')
                if speech_who is not None and characterdict['id'] in speech_who:
                    # Speaker text is contained in lines with <l> tags
                    for line in speech.findall('.//tei:l', ns):
                        if line.text is not None:  # Check if line.text is not None
                            tokens.append(line.text)
                            lemmas.append(lemmatize(line.text))
            characterdict['tokens'] = tokens
            tokens_length = len(tokens)
            characterdict['tokens_length'] = tokens_length
            characterdict['lemmas'] = lemmas
            characterlist.append(characterdict)

    return characterlist

def read_play_file(xml_file):
    tree = ET.parse(xml_file)  # Parse the XML file once
    root = tree.getroot()
    
    characters = get_speaker_tokens(root)  # Pass the parsed root to the function
    
    # Extract title information from the XML
    # the title is the text of <title type="sub">
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    play_title_element = root.find('.//tei:title[@type="main"]', ns)
    play_title = play_title_element.text
    print(play_title)


    # Create an empty DataFrame
    df = pd.DataFrame(columns=['play_title', 'character_id', 'character_name', 'character_gender', 'character_tokens', 'character_lemmas'])

    character_data = []

    # Add rows for each character to the DataFrame
    for character in characters:
        character_data.append({
            'play_title': play_title,
            'character_id': character['id'],
            'character_name': character['Name'],
            'character_gender': character['gender'],
            'words_spoken': character['tokens_length'],  # This is the number of words spoken by the character
            'character_tokens': ' '.join(character['tokens']),  # Combine tokens into a single string
            'character_lemmas': character['lemmas']
        })

    df = pd.DataFrame(character_data)
    print(type(df))
    return df

# Now read each file in the results directory and add the words spoken by each character to the dataframe in the column 'tokens'
# with open('/Users/allisonkeith/VSCode_folder/projectcalderon/calderonplays/results/a-secreto-agravio-secreta-venganza.xml', 'r') as f:
#     df = read_play_file(f)
#     print(df.head())

#creat a dataframe with all the plays
maindf = pd.DataFrame(columns=['id','play_title','character_id','character_name','character_gender','character_tokens','character_lemmas'])

dfs = []

for file in os.listdir(input_directory):
    if file.endswith('.xml'):
        with open(os.path.join(input_directory, file), 'r') as f:
            
            df = read_play_file(f)
            df['id'] = file
            print(type(df))
            print(df.shape)
            dfs.append(df)
            result = pd.concat(dfs)


print(result.head())
            
result.to_csv(output_CSV, index=False)


"""
This code is to add a column to the dataframe that contains the lines spoken by each character.
the dataframe will then contain the following columns:
id,genre,character_id,character_name,character_gender,tokens_spoken,character_tokens,character_lemmas,character_utterances,play_title
which will allow analysis of gender prediction based on
1. sentences
2. utterances
3. all lines
"""

import pandas as pd
import os
import re
import xml.etree.ElementTree as ET

# input_directory = '/Users/allisonkeith/VSCode_folder/projectcalderon/calderonplays/results'
# output_CSV = '/Users/allisonkeith/VSCode_folder/projectcalderon/wp1-semantic-analysis/character_analysis/gender_analysis/gender_classifier/character_utterances.csv'


# # Function that produces a list of dictionaries with character information from the xml file
# def get_speaker_tokens(root):
#     characterlist = []
#     ns = {'tei': 'http://www.tei-c.org/ns/1.0', 'xml': 'http://www.w3.org/XML/1998/namespace'}

#     for person in root.findall('.//tei:person', ns):
#         name = person.find('.//tei:persName', ns).text
#         xml_id = person.get('{http://www.w3.org/XML/1998/namespace}id')
#         if xml_id is not None:
#             characterdict = {'id': xml_id, 'Name': name, 'gender': person.get('sex'), 'tokens_length': '', 'tokens': [], 'utterances': [], 'sentences': []}
#             utterances = []
#             tokens = []
#             for speech in root.findall('.//tei:sp', ns):
#                 speech_who = speech.get('who')
#                 utterance = ""
#                 if speech_who is not None and characterdict['id'] in speech_who:
#                     # Speaker text is contained in lines with <l> tags
#                     for line in speech.findall('.//tei:l', ns):
#                         if line.text is not None:  # Check if line.text is not None
#                             utterance += line.text + " "
#                             #print("PRINTING LINE.TEXT")
#                             #print(line.text)
#                 utterances.append(utterance)
#                 #remove empty strings from utterances
#                 utterances = [x for x in utterances if x]
                


#             characterdict['tokens'] = tokens
#             tokens_length = len(tokens)
#             characterdict['tokens_length'] = tokens_length
#             characterdict['utterances'] = utterances
#             #print("PRINTING UTTERANCES")
#             #print(utterances)
#             sentences = []
#             for utterance in utterances:
#                 if utterance is not None:
#                     sentence = re.split(r'[.¡!¿?]+', utterance)
#                     sentences.append(sentence)

#             characterdict['sentences'] = sentences
#             print("PRINTING SENTENCES")
#             print(sentences)
                    

#             characterlist.append(characterdict)

#     return characterlist

# def read_play_file(xml_file):
#     tree = ET.parse(xml_file)  # Parse the XML file once
#     root = tree.getroot()
    
#     characters = get_speaker_tokens(root)  # Pass the parsed root to the function
    
#     # Extract title information from the XML
#     # the title is the text of <title type="sub">
#     ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
#     play_title_element = root.find('.//tei:title[@type="main"]', ns)
#     play_title = play_title_element.text
#     print(play_title)


#     # Create an empty DataFrame
#     df = pd.DataFrame(columns=['play_title', 'character_id', 'character_name', 'character_gender', 'character_tokens','character_utterances','character_sentences'])

#     character_data = []

#     # Add rows for each character to the DataFrame
#     for character in characters:
#         character_data.append({
#             'play_title': play_title,
#             'character_id': character['id'],
#             'character_name': character['Name'],
#             'character_gender': character['gender'],
#             'words_spoken': character['tokens_length'],  # This is the number of words spoken by the character
#             'character_tokens': ' '.join(character['tokens']),  # Combine tokens into a single string
#             'character_utterances': character['utterances'],
#             'character_sentences': character['sentences']

#         })

#     df = pd.DataFrame(character_data)
#     print(type(df))
#     return df

# # Now read each file in the results directory and add the words spoken by each character to the dataframe in the column 'tokens'
# # with open('/Users/allisonkeith/VSCode_folder/projectcalderon/calderonplays/results/a-secreto-agravio-secreta-venganza.xml', 'r') as f:
# #     df = read_play_file(f)
# #     print(df.head())

# #creat a dataframe with all the plays
# maindf = pd.DataFrame(columns=['id','play_title','character_id','character_name','character_gender','character_tokens', 'character_utterances', 'character_sentences'])

# dfs = []

# for file in os.listdir(input_directory):
#     if file.endswith('.xml'):
#         with open(os.path.join(input_directory, file), 'r') as f:
            
#             df = read_play_file(f)
#             df['id'] = file
#             print(type(df))
#             print(df.shape)
#             dfs.append(df)
#             result = pd.concat(dfs)


# print(result.head())
            
# result.to_csv(output_CSV, index=False)


################################
#since adding the the tokens didn't work, I'll merge the two dataframes
################################
# utterances_csv = '/Users/allisonkeith/VSCode_folder/projectcalderon/wp1-semantic-analysis/character_analysis/gender_analysis/gender_classifier/character_utterances.csv'
# tokens_csv = '/Users/allisonkeith/VSCode_folder/projectcalderon/wp1-semantic-analysis/character_analysis/gender_analysis/topic_modeling/characters_with_clean_tokens.csv'


# utterances_df = pd.read_csv(utterances_csv)
# tokens_df = pd.read_csv(tokens_csv)

# combined_df = utterances_df.combine_first(tokens_df)
# print(combined_df.head(20))
# for header in combined_df.columns:
#     print(header)

# print(combined_df.head())


# combined_df.to_csv('/Users/allisonkeith/VSCode_folder/projectcalderon/wp1-semantic-analysis/character_analysis/gender_analysis/gender_classifier/gender_prediction_data.csv', index = False)

##############################################
#try a different method
##############################################


# #concatenate all the lists in the column 'character_utterances' which is a list of strings into a single string 
# df = pd.read_csv(utterances_csv)

# tokens = ''

# import ast

# df['character_utterances'] = df['character_utterances'].apply(ast.literal_eval)
# df['character_sentences'] = df['character_sentences'].apply(ast.literal_eval)


# df['tokens'] = df['character_utterances'].apply(lambda x: ' '.join(x))


# df.to_csv('/Users/allisonkeith/VSCode_folder/projectcalderon/wp1-semantic-analysis/character_analysis/gender_analysis/gender_classifier/gender_prediction_data.csv', index = False)

############################################## 
#now count the number of words in df['tokens'] and add that to the dataframe
##############################################
utterances_csv = '/Users/allisonkeith/VSCode_folder/projectcalderon/wp1-semantic-analysis/character_analysis/gender_analysis/gender_classifier/gender_prediction_data.csv'

df = pd.read_csv(utterances_csv)

print(df.loc[[2521]])
print(df['tokens'][0])
print(type(df['tokens'][0]))

def string_length(string):
    return len(str(string).split())

df['tokens_length'] = df['tokens'].apply(string_length)

# for index, row in df.iterrows():
#     row['tokens_length'] = 0
#     row['tokens_length'] = len(str(row['tokens']).split())



print(df.columns)
print(df.head()[:5])


df.to_csv('/Users/allisonkeith/VSCode_folder/projectcalderon/wp1-semantic-analysis/character_analysis/gender_analysis/gender_classifier/gender_prediction_data.csv', index = False)
