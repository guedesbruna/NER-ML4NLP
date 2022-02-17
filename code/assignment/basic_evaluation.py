# !pip install nose

import sys
import pandas as pd
from collections import defaultdict, Counter
from nose.tools import assert_equal
import numpy as np
import sys


def extract_annotations(inputfile, annotationcolumn, delimiter='\t'):
    '''
    This function extracts annotations represented in the conll format from a file
    
    :param inputfile: the path to the conll file
    :param annotationcolumn: the name of the column in which the target annotation is provided
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type inputfile: string
    :type annotationcolumn: string
    :type delimiter: string
    :returns: the annotations as a list
    '''
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    conll_input = pd.read_csv(inputfile, sep=delimiter, on_bad_lines='skip') #, quotechar=delimiter)
    annotations = conll_input[annotationcolumn].tolist()
    return annotations



def obtain_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output
    
    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations
    
    :returns: a countainer providing the counts for each predicted and gold class pair
    '''
    evaluation_counts = defaultdict(Counter)

    for gold_annotation, machine_annotation in zip(goldannotations, machineannotations):
        evaluation_counts[gold_annotation][machine_annotation] += 1
        
    return evaluation_counts  

    # TIP on how to get the counts for each class
    # https://stackoverflow.com/questions/49393683/how-to-count-items-in-a-nested-dictionary, last accessed 22.10.2020
    evaluation_counts = defaultdict(Counter)
    

def provide_confusion_matrix(evaluation_counts):
    '''
    Read in the evaluation counts and provide a confusion matrix for each class
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :prints out a confusion matrix'''
    
    confusion_matrix = pd.DataFrame.from_dict({i: evaluation_counts[i] for i in evaluation_counts.keys()}, orient='index')
    
    return confusion_matrix
    

def calculate_precision_recall_fscore(evaluation_counts):
    '''
    Calculate precision recall and fscore for each class and return them in a dictionary
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :returns the precision, recall and f-score of each class in a container

        Concepts:    
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2* (precision * recall) / (precision + recall)''' 

    confm = provide_confusion_matrix(evaluation_counts)
    sum_within_row = confm.sum(axis=1)
    sum_within_column = confm.sum(axis=0)
    total = confm.sum()
    
    acc = []
    prec = []
    recall = []
    f1 = []
    for i in range(4):
        acc.append((confm[i][i] + (total - sum_within_column[i] - sum_within_row[i] + confm[i][i]))/ total) 
        prec.append(confm[i][i] / sum_within_column[i])
        recall.append(confm[i][i] / sum_within_row[i])
        f1.append((2* (prec[i] * recall[i])) / (prec[i] + recall[i]))
    
    return acc, prec, recall, f1



def carry_out_evaluation(gold_annotations, systemfile, systemcolumn, delimiter='\t'):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)
    
    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')
    
    returns evaluation information for this specific system
    '''
    system_annotations = extract_annotations(systemfile, systemcolumn, delimiter)
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)
    
    return evaluation_outcome


def provide_output_tables(evaluations):
    '''
    Create tables based on the evaluation of various systems
    
    :param evaluations: the outcome of evaluating one or more systems
    '''
    #https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    evaluations_pddf = pd.DataFrame.from_dict({(i,j): evaluations[i][j]
                                              for i in evaluations.keys()
                                              for j in evaluations[i].keys()},
                                             orient='index')
    print(evaluations_pddf)
    print(evaluations_pddf.to_latex())


def run_evaluations(goldfile, goldcolumn, systems):
    '''
    Carry out standard evaluation for one or more system outputs
    
    :param goldfile: path to file with goldstandard
    :param goldcolumn: indicator of column in gold file where gold labels can be found
    :param systems: required information to find and process system output
    :type goldfile: string
    :type goldcolumn: integer
    :type systems: list (providing file name, information on tab with system output and system name for each element)
    
    :returns the evaluations for all systems
    '''
    evaluations = {}
    #not specifying delimiters here, since it corresponds to the default ('\t')
    gold_annotations = extract_annotations(goldfile, goldcolumn)
    for system in systems:
        sys_evaluation = carry_out_evaluation(gold_annotations, system[0], system[1])
        evaluations[system[2]] = sys_evaluation
    return evaluations


def identify_evaluation_value(system, class_label, value_name, evaluations):
    '''
    Return the outcome of a specific value of the evaluation
    
    :param system: the name of the system
    :param class_label: the name of the class for which the value should be returned
    :param value_name: the name of the score that is returned
    :param evaluations: the overview of evaluations
    
    :returns the requested value
    '''
    return evaluations[system][class_label][value_name]


def create_system_information(system_information):
    '''
    Takes system information in the form that it is passed on through sys.argv or via a settingsfile
    and returns a list of elements specifying all the needed information on each system output file to carry out the evaluation.
    
    :param system_information is the input as from a commandline or an input file
    '''
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    systems_list = [system_information[i:i + 3] for i in range(0, len(system_information), 3)]
    return systems_list


## My approach and functions used 


def initialize_feat_lab(train, dev_spacy_stan):
    df_train = pd.read_table(train) 
    df_dev = pd.read_table('../../data/preprocessed/conll2003.dev-preprocessed.conll')
    df_spacy_stan = pd.read_table(dev_spacy_stan)

    dev_gold = df_dev['NER'].tolist()
    spacy_stan_lab = df_spacy_stan['NER2'].tolist()

    return spacy_stan_lab, dev_gold

def obtain_counts(goldannotations, machineannotations):

    evaluation_counts = defaultdict(Counter)

    for gold_annotation, machine_annotation in zip(goldannotations, machineannotations):
        evaluation_counts[gold_annotation][machine_annotation] += 1
        
    return evaluation_counts  

def provide_confusion_matrix(evaluation_counts):
    ''' Read in the evaluation counts and provide a confusion matrix for each class
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts'''
    
    confusion_matrix = pd.DataFrame.from_dict({i: evaluation_counts[i] for i in evaluation_counts.keys()}, orient='index')
    confusion_matrix = confusion_matrix.reindex(sorted(confusion_matrix.columns), axis=1)
    confusion_matrix = confusion_matrix.reindex(sorted(confusion_matrix.columns), axis=0)
    confusion_matrix = confusion_matrix.fillna(0)
    confusion_matrix = confusion_matrix.round(0).astype(int)

    return confusion_matrix

def create_table_pre_rec_f1(prec, recall, f1, macro_prec, macro_recall, macro_f1):
    data = [{'Precision':prec[0], 'Recall': recall[0], 'F1-score': f1[0]},
            {'Precision':prec[1], 'Recall': recall[1], 'F1-score': f1[1]},
            {'Precision':prec[2], 'Recall': recall[2], 'F1-score': f1[2]},
            {'Precision':prec[3], 'Recall': recall[3], 'F1-score': f1[3]},
            {'Precision':prec[4], 'Recall': recall[4], 'F1-score': f1[4]},
            {'Precision':prec[5], 'Recall': recall[5], 'F1-score': f1[5]},
            {'Precision':prec[6], 'Recall': recall[6], 'F1-score': f1[6]},
            {'Precision':prec[7], 'Recall': recall[7], 'F1-score': f1[7]},
            {'Precision':prec[8], 'Recall': recall[8], 'F1-score': f1[8]},
            {'Precision':macro_prec, 'Recall': macro_recall, 'F1-score': macro_f1}
            ]
    
    # Creates DataFrame.
    df_summary = pd.DataFrame(data, index =['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O', 'macro'])
    df_summary = df_summary.round(3)
    return df_summary

def calculate_precision_recall_fscore(evaluation_counts):
    ''' Concepts:    
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2* (precision * recall) / (precision + recall)''' 

    confm = provide_confusion_matrix(evaluation_counts)
    sum_within_row = confm.sum(axis=1) 
    sum_within_column = confm.sum(axis=0) 
    
    prec = []
    recall = []
    f1 = []
    for i in range(len(confm)):
        prec.append(confm.iloc[i, i] / sum_within_column[i])
        recall.append(confm.iloc[i, i] / sum_within_row[i])
        f1.append((2* (prec[i] * recall[i])) / (prec[i] + recall[i]))
    
    macro_prec = sum(prec)/len(prec)
    macro_recall = sum(recall)/len(recall)
    macro_f1 = sum(f1)/len(f1)

    table = create_table_pre_rec_f1(prec, recall, f1, macro_prec, macro_recall, macro_f1)
    return macro_prec, macro_recall, macro_f1, table


def results(label_gold, label_model):
    eval_counts = obtain_counts(label_gold, label_model)
    confm = provide_confusion_matrix(eval_counts)
    prec,recall, f1, table = calculate_precision_recall_fscore(eval_counts)
    return table, confm



def main(argv=None):

    if argv is None:
        argv = sys.argv

    trainingfile = argv[1] # path to training file 
    outputfile = argv[2] # path including name of where file should be generated and how it should be called  
    stan_spacy = argv[3] # name that will be added to file header: here 'stanford' or 'spacy'

    # df_train, df_spacy_stan, train_feat, spacy_stan_feat, train_gold, spacy_stan_lab, df_dev, dev_feat, dev_gold = initialize_feat_lab(argv[1], argv[2])
    spacy_stan_lab, dev_gold = initialize_feat_lab(trainingfile, outputfile)
    
    table, confm = results(dev_gold, spacy_stan_lab)

    table.to_csv('./results/'+ stan_spacy + '_pre_rec_f1.conll', sep='\t')
    confm.to_csv('./results/'+ stan_spacy + '_confm.conll', sep='\t')
    


if __name__ == '__main__':
    main()


#python3 basic_evaluation.py '../../data/preprocessed/conll2003.train-preprocessed.conll' '../../models/output/output_system.SVM.conll' conll_SVM
#python3 basic_evaluation.py '../../data/preprocessed/conll2003.train-preprocessed.conll' '../../models/output/output_system.NB.conll' conll_NB
#python3 basic_evaluation.py '../../data/preprocessed/conll2003.train-preprocessed.conll' '../../models/output/output_system.logreg.conll' conll_logreg




