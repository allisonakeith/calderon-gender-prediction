"""
This code is to add a column to the dataframe that contains the lines spoken by each character.
the dataframe will then contain the following columns:
id,genre,character_id,character_name,character_gender,tokens_spoken,character_tokens,character_utterances,play_title
which will allow analysis of gender prediction based on
1. scene text
2. utterances
3. all lines
"""

import pandas as pd
import os
import re
import xml.etree.ElementTree as ET

input_directory = 'calderon-gender-prediction/tei_plays'
output_CSV = 'calderon-gender-prediciton/all_characters.csv'


# Function that produces a list of dictionaries with character information from the xml file
def get_speaker_tokens(root):
    characterlist = []
    ns = {'tei': 'http://www.tei-c.org/ns/1.0', 'xml': 'http://www.w3.org/XML/1998/namespace'}

    for person in root.findall('.//tei:person', ns):
        name = person.find('.//tei:persName', ns).text
        xml_id = person.get('{http://www.w3.org/XML/1998/namespace}id')
        if xml_id is not None:
            characterdict = {'id': xml_id, 'Name': name, 'gender': person.get('sex'), 'tokens_length': '', 'tokens': '', 'scenes': [], 'num_sences': '' , 'utterances': []}
            utterances = []
            tokens = ''
            for speech in root.findall('.//tei:sp', ns):
                speech_who = speech.get('who')
                utterance = ""
                if speech_who is not None and characterdict['id'] in speech_who:
                    # Speaker text is contained in lines with <l> tags
                    for line in speech.findall('.//tei:l', ns):
                        if line.text is not None:  # Check if line.text is not None
                            utterance += line.text + " "

                            # Remove new line characters
                            utterance = re.sub('\n', ' ', utterance)   

                            utterance = re.sub(' +', ' ', utterance) 

                    utterances.append(utterance)
                 
                    tokens += utterance + " "     
                    #remove empty strings from utterances
                    #utterances = [x for x in utterances if x]
            tokens = re.sub(' +', ' ', tokens.strip())


            #Now make the tokens:


            characterdict['tokens'] = tokens
            tokens_length = len(tokens.split())
            characterdict['tokens_length'] = tokens_length


            ###############################################
            # I also want to make a list of lines spoken by each character for each scene


            scenes = []
            for act in root.findall('.//tei:div[@type="act"]', ns):
                act_number = act.get('n')
                # for scene in root.findall('.//tei:div[@type="scene"]', ns):
                for scene in act.findall('.//tei:div[@type="scene"]', ns):
                    scene_number = scene.get('n')
                    
                    #add act and scene number    

                    words_in_scene = ""
                    for speech in scene.findall('.//tei:sp', ns):
                        speech_who = speech.get('who')
                        
                        if speech_who is not None and characterdict['id'] in speech_who:
                            # Speaker text is contained in lines with <l> tags
                            for line in speech.findall('.//tei:l', ns):
                                if line.text is not None:
                                    scene_lines = line.text
                                    words_in_scene += "" + scene_lines

                                    #remove extra spaces and new line characters
                                    words_in_scene = re.sub('\n', ' ', words_in_scene) 
                                    words_in_scene = re.sub(' +', ' ', words_in_scene)

                    if words_in_scene != '':
                        scene_text_with_act = {f"{act_number}:{scene_number}" : words_in_scene}
                        scenes.append(scene_text_with_act)
                        
                        
                characterdict['scenes'] = scenes

                num_scenes = len(scenes)
                characterdict['num_scenes'] = num_scenes
                            


            ###############################################

            #count the number of utterances:
            characterdict['num_utterances'] = len(utterances)

            characterdict['utterances'] = utterances


            characterlist.append(characterdict)

    return characterlist

def read_play_file(xml_file):
    tree = ET.parse(xml_file)  # Parse the XML file once
    root = tree.getroot()

    ### Only do this for comedias
    genre = tree.find(".//{http://www.tei-c.org/ns/1.0}textClass/{http://www.tei-c.org/ns/1.0}keywords/{http://www.tei-c.org/ns/1.0}term[@type='genreTitle']")

    if genre is not None:
        genre = genre.text
        print(genre)
    characters = get_speaker_tokens(root)  # Pass the parsed root to the function
        
    # Extract title information from the XML
    # the title is the text of <title type="sub">
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    play_title_element = root.find('.//tei:title[@type="main"]', ns)
    play_title = play_title_element.text
    print(play_title)

    character_data = []

    # Add rows for each character to the DataFrame

    for character in characters:
        character_data.append({
            'play_title': play_title,
            'genre': genre,
            'character_id': character['id'],
            'character_name': character['Name'],
            'character_gender': character['gender'],
            'words_spoken': character['tokens_length'],  # This is the number of words spoken by the character
            'tokens': character['tokens'],
            'num_utterances': character['num_utterances'],
            'utterances': character['utterances'],
            'scenes': character['scenes'],
            'num_scenes': character['num_scenes']

        })

    df = pd.DataFrame(character_data)

    return df

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


###################################################
            
result.to_csv(output_CSV, index=False)