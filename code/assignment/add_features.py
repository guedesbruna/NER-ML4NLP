import csv
import pandas as pd
import nltk
import sys

def read_in_conll_file(conll_file, delimiter='\t'):
    '''Read in conll file and return structured object
    inputs:conll_file: path to conll_file
    outputs:returns structured representation of information included in conll file'''

    my_conll = open(conll_file, 'r')
    conll_as_csvreader = csv.reader(my_conll, delimiter=delimiter)
    return conll_as_csvreader

def starts_with_capital(conll_file):
    '''Return a list of items specifying if the token starts with a capital letter or not'''

    conll_object = read_in_conll_file(conll_file)
    
    all_tokens = []
    capitals = []
    
    for index, row in enumerate(conll_object): 
        if index == 0: 
            continue
        if len(row) > 0:
            token = row[0]
            all_tokens.append(token)
     
    for token in all_tokens:
        if token.isupper():
            capitals.append(1) 
        else:
            capitals.append(0) 
    
    return capitals

def get_previous_and_following_token(conll_file):
    '''Return two lists: one with the previous tokens and one with the following tokens'''

    conll_object = read_in_conll_file(conll_file)
    
    all_tokens = []
    previous_tokens = []
    following_tokens = []
    
    for index, row in enumerate(conll_object): 
        if index == 0: 
            continue
        if len(row) > 0:
            token = row[0]
            all_tokens.append(token)
    
    previous = ' '
    following = ' '

    # Creates two lists one for previous and another for following tokens
    for index, token in enumerate(all_tokens):
        
        if index > 0:
            previous = all_tokens[index - 1]
        previous_tokens.append(previous)
        
        if index < (len(all_tokens) - 1):
            following = all_tokens[index + 1]
        following_tokens.append(following)
    
    return previous_tokens, following_tokens

def main():

    args = sys.argv
    trainingfile = args[1]
    evaluationfile = args[2]
    
    df_training = pd.read_csv(trainingfile, sep='\t')
    df_test = pd.read_csv(evaluationfile, sep='\t')

    # Adding new features (previous tokens, following tokens and capitalization) to training file
    previous_tokens_training, following_tokens_training = get_previous_and_following_token(trainingfile)
    capitals_training = starts_with_capital(trainingfile)

    df_training['previous'] = previous_tokens_training
    df_training['following'] = following_tokens_training
    df_training['capitals'] = capitals_training

    df_training.to_csv(trainingfile.replace('.conll', '_with_features.conll'), sep='\t', index=False) 

    # Adding new features (previous tokens, following tokens and capitalization)to file with gold data
    previous_tokens_gold, following_tokens_gold = get_previous_and_following_token(evaluationfile)
    capitals_gold = starts_with_capital(evaluationfile)

    df_test['previous'] = previous_tokens_gold
    df_test['following'] = following_tokens_gold
    df_test['capitals'] = capitals_gold

    df_test.to_csv(evaluationfile.replace('.conll', '_with_features.conll'), sep='\t', index=False) 

    
if __name__ == '__main__':
    main()

#run the below lines on the terminal: 

#python3 add_features.py '../../data/preprocessed/conll2003.train-preprocessed.conll' '../../data/preprocessed/conll2003.dev-preprocessed.conll'
#python3 add_features.py '../../data/preprocessed/conll2003.train-preprocessed.conll' '../../data/preprocessed/stanford_out.dev-preprocessed.conll'
#python3 add_features.py '../../data/preprocessed/conll2003.train-preprocessed.conll' '../../data/preprocessed/spacy_out.dev-preprocessed.conll'