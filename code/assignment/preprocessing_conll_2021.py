import csv
from typing import List, Dict


def matching_tokens(conll1: List, conll2: List) -> bool:
    '''
    Check whether the tokens of two conll files are aligned
    
    :param conll1: tokens (or full annotations) from the first conll file
    :param conll2: tokens (or full annotations) from the first conll file
    
    :returns boolean indicating whether tokens match or not
    '''
    for i, row in enumerate(conll1):
        row2 = conll2[i]
        if row[0] != row2[0]:
            return False
    
    return True



def read_in_conll_file(conll_file: str, delimiter: str = '\t'):
    '''
    Read in conll file and return structured object
    
    :param conll_file: path to conll_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll
    
    :returns List of splitted rows included in conll file
    '''
    conll_rows = []
    with open(conll_file, 'r') as my_conll:
        for line in my_conll:
            row = line.strip("\n").split(delimiter)
            if len(row) == 1:
                conll_rows.append([""]*rowlen)
            else:
                rowlen = len(row)
                conll_rows.append(row)
    return conll_rows



def alignment_okay(conll1: str, conll2: str) -> bool:
    '''
    Read in two conll files and see if their tokens align
    '''
    my_first_conll = read_in_conll_file(conll1)
    my_second_conll = read_in_conll_file(conll2)
    
    return matching_tokens(my_first_conll, my_second_conll)
    


def get_predefined_conversions(conversion_file: str) -> Dict:
    '''
    Read in file with predefined conversions and return structured object that maps old annotation to new annotation
    
    :param conversion_file: path to conversion file
    
    :returns object that maps old annotations to new ones
    '''
    conversion_dict = {}
    my_conversions = open(conversion_file, 'r')
    conversion_reader = csv.reader(my_conversions, delimiter='\t')
    for row in conversion_reader:
        conversion_dict[row[0]] = row[1]
    return conversion_dict



def create_converted_output(conll_rows: List, annotation_identifier: int, conversions: Dict, outputfilename: str, delimiter: str = '\t'):
    '''
    Check which annotations need to be converted for the output to match and convert them
    
    :param conll_rows: rows with conll annotations
    :param annotation_identifier: indicator of how to find the annotations in the object (index)
    :param conversions: pointer to the conversions that apply. This can be external (e.g. a local file with conversions) or internal (e.g. prestructured dictionary). In case of an internal object, you probably want to add a function that creates this from a local file.
    
    '''
    with open(outputfilename, 'w') as outputfile:
        for row in conll_rows:
            annotation = row[annotation_identifier]
            if annotation in conversions:
                row[annotation_identifier] = conversions.get(annotation)
            if row[0] == "":
                outputfile.write("\n")
            else:
                outputfile.write(delimiter.join(row)+"\n")




def preprocess_files(conll1: str, conll2: str, column_identifiers: List, conversions: Dict):
    '''
    Guides the full process of preprocessing files and outputs the modified files.
    
    :param conll1: path to the first conll input file
    :param conll2: path to the second conll input file
    :param column_identifiers: object providing the identifiers for target column
    :param conversions: path to a file that defines conversions
    '''
    if alignment_okay(conll1, conll2):
        conversions = get_predefined_conversions(conversions)
        my_first_conll = read_in_conll_file(conll1)
        my_second_conll = read_in_conll_file(conll2)
        create_converted_output(my_first_conll, column_identifiers[0], conversions, conll1.replace('.conll','-preprocessed0.conll'))
        create_converted_output(my_second_conll, column_identifiers[1], conversions, conll2.replace('.conll','-preprocessed0.conll'))
    else:
        print(conll1, conll2, 'do not align')



preprocess_files('../../data/spacy_out.dev.conll','../../data/conll2003.dev.conll', [2,3],'../settings/conversions.tsv')
preprocess_files('../../data/stanford_out.dev.conll','../../data/conll2003.dev.conll', [3,3],'../settings/conversions.tsv')


# Extended preprocessing to align labels


import pandas as pd
import numpy as np

df = pd.read_table('../../data/conll2003.dev-preprocessed0.conll', header = None,  sep='\t', on_bad_lines='skip', quotechar='\t')
df_train = pd.read_table('../../data/conll2003.train.conll', header = None,  sep='\t', on_bad_lines='skip', quotechar='\t')
df_spacy = pd.read_table('../../data/spacy_out.dev-preprocessed0.conll', header = None,  sep='\t', on_bad_lines='skip', quotechar='\t')
df_stan = pd.read_table('../../data/stanford_out.dev-preprocessed0.conll', header = None,  sep='\t', on_bad_lines='skip', quotechar='\t')

df = df.rename(columns={0: "token", 1: "POS", 2:"chunk", 3:"NER"})
df_train = df_train.rename(columns={0: "token", 1: "POS", 2:"chunk", 3:"NER"})
df_spacy = df_spacy.rename(columns={0: "token", 1: "NER0", 2:"type_ent"})
df_stan = df_stan.drop(columns = [1,4,5])
df_stan = df_stan.rename(columns={0: "token", 2: "POS", 3:"NER0"})


# rename everything that not person, org, loc as MISC
df_spacy =df_spacy.replace({'type_ent': ['DATE', 'NORP', 'CARDINAL',
       'TIME', 'FAC', 'ORDINAL', 'LANGUAGE', 'LAW', 'MONEY', 'EVENT', 'O',
       'PERCENT', 'PRODUCT', 'I-PER', 'QUANTITY', 'WORK_OF_ART']}, 'MISC')
df_spacy =df_spacy.replace({'type_ent': 'LOC'}, 'LOCATION')

df_spacy['type_ent'].unique()


# create a list of our conditions spacy
conditions = [
     (df_spacy['NER0'] == 'B') & (df_spacy['type_ent'] == 'PERSON'),
     (df_spacy['NER0'] == 'B') & (df_spacy['type_ent'] == 'ORG'),
     (df_spacy['NER0'] == 'B') & (df_spacy['type_ent'] == 'LOCATION'),
     (df_spacy['NER0'] == 'I') & (df_spacy['type_ent'] == 'PERSON'),
     (df_spacy['NER0'] == 'I') & (df_spacy['type_ent'] == 'ORG'),
     (df_spacy['NER0'] == 'I') & (df_spacy['type_ent'] == 'LOCATION'),
     (df_spacy['NER0'] == 'B') & (df_spacy['type_ent'] == 'MISC'),
     (df_spacy['NER0'] == 'I') & (df_spacy['type_ent'] == 'MISC'),
     (df_spacy['NER0'] == 'O')]


# create a list of the values we want to assign for each condition
values = ['B-PER', 'B-ORG', 'B-LOC', 'I-PER', 'I-ORG', 'I-LOC', 'B-MISC','I-MISC','O']

# create a new column and use np.select to assign values to it using our lists as arguments
df_spacy['NER2'] =  np.select(conditions, values)


# create a list of our conditions stanford
conditions_stan = [
     (df_stan['NER0'] == 'PERSON'),
     (df_stan['NER0'] == 'ORG'),
     (df_stan['NER0'] == 'LOCATION'),
     (df_stan['NER0'] == 'O')]


# create a list of the values we want to assign for each condition
values_stan = ['B-PER', 'B-ORG', 'B-LOC', 'O']

# create a new column and use np.select to assign values to it using our lists as arguments
df_stan['NER2'] =  np.select(conditions_stan, values_stan)

df_stan = df_stan.replace({'NER2': '0'}, 'B-MISC')


for i in range(1, len(df_stan)):

    if (df_stan['NER2'][i] == 'B-ORG') and (df_stan['NER2'][i-1] == 'B-ORG'):
        df_stan['NER2'][i] = 'I-ORG'
    elif (df_stan['NER2'][i] == 'B-PER') and (df_stan['NER2'][i-1] == 'B-PER'):
        df_stan['NER2'][i] = 'I-PER'
    elif (df_stan['NER2'][i] == 'B-LOC') and (df_stan['NER2'][i-1] == 'B-LOC'):
        df_stan['NER2'][i] = 'I-LOC'
    elif (df_stan['NER2'][i] == 'B-MISC') and (df_stan['NER2'][i-1] == 'B-MISC'):
        df_stan['NER2'][i] = 'I-MISC'


# generate preprocessed files in conll format
df.to_csv('../../data/preprocessed/conll2003.dev-preprocessed.conll', sep='\t', index=False)
df.to_csv('../../data/preprocessed/conll2003.train-preprocessed.conll', sep='\t', index=False)
df_stan.to_csv('../../data/preprocessed/stanford_out.dev-preprocessed.conll', sep='\t', index=False)
df_spacy.to_csv('../../data/preprocessed/spacy_out.dev-preprocessed.conll', sep='\t', index=False)



