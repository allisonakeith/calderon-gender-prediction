# calderon-gender-prediction

This package takes as input a directory of plays in tei-xml file format and runs them through a pipeline that uses LLMs to predict the gender of characters in the works based on different length of dialogue input. The aim is to determine if characters in the works follow strict gender conventions.

The files are listed here in the order they should be run:
### File Structure

1. preprocessing

   1. tei2csv_scene_nums.py
      Reads the plays and creates a csv file where each character is a row
   2. masking.py
      Masks character names, locations, and uncommon words in the plays

2. pipeline

   1. model.py
      Contains the LLM classification pipeline - split.py splits the data into train, test, and validation sets - interpret.py contains the attribution function - test_train.py contains the training and testing functions

3. analysis

    1. analyze_gender_prediction_results.ipynb
        Contains the functions to analyze the results of the pipeline - this is where we get precision, recall, and f1 scores
    2. interpret_attributions.ipynb
        Contains the functions to interpret the attributions of the model - here we see the words with the most extreme attributions
        - visualize.ipynb
            I created a personalized visualization function based on the visualizaiton from TransformersInterpret 
    3. scene_level.ipynb
        Allows us to anaylze the character results at the scene level. Specifically, we use this to analyze the predictions of cross-dressing characters.


    
