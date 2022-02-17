from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import sys
import numpy as np
from basic_evaluation import results ,initialize_feat_lab


def extract_features_and_labels(trainingfile):
    '''extract column of features (data) and labels (target) from training file and outputs both variables in a list format.
    outputs: data a list of dictionaries of format [{'token': ',token,POS,chunk,NER'}] and 
    outputs: targets being a list of the values of the data in fotmat [',token,POS,chunk,NER']'''
    
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token':token}
                data.append(feature_dict)
                #gold is in the last column
                targets.append(components[-1])
    return data, targets


def extract_features(inputfile):
    '''extracts column of features only from input file and returns a list of dictionaries
    output format: [{'token': ',token,POS,chunk,NER'}]'''
   
    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                feature_dict = {'token':token}
                data.append(feature_dict)
    return data


def create_classifier(train_features, train_targets):
   '''Creates a Logistic regression classifier that takes the list of features as the X and the list of labels as the Y.
   It fits the model by using a vectorized transformation of the features that now acquire the vector format of (0, 0)	1.0
   outputs the model for the Logistic regression and vec which is a function that transforms lists to vectors'''

   logreg = LogisticRegression(max_iter=1000)
   vec = DictVectorizer()
   features_vectorized = vec.fit_transform(train_features)
   model = logreg.fit(features_vectorized, train_targets)
   
   return model, vec


def classify_data(model, vec, inputdata, outputfile):
    '''Extracts features from input data, transforms this list it into a vector. Input the vector into the model to fit a Logistic Regression model.
    Then it opens an output file and for each line in the original input file it writes the token, tab, the prediction of the model. 
    outputs the file with the predictions.'''
    
    features = extract_features(inputdata)
    features = vec.transform(features)
    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()


def main(argv=None):
    
    if argv is None:
        argv = sys.argv
    
    trainingfile = argv[1] # path to training file 
    inputfile = argv[2]    # path to dev file that will be used to generate output
    outputfile = argv[3]   # path including name of where file should be generated and how it should be called
    stan_spacy = argv[4]   # name that will be added to file header
    
    training_features, gold_labels = extract_features_and_labels(trainingfile) 
    ml_model, vec = create_classifier(training_features, gold_labels)
    classify_data(ml_model, vec, inputfile, outputfile)

    df = pd.read_table(outputfile)
    df = df.set_axis([*df.columns[:-1], 'NER2'], axis=1, inplace=False)
    df.to_csv(outputfile, sep='\t')

    spacy_stan_lab, dev_gold = initialize_feat_lab(argv[1], outputfile) #argv[3])

    table, confm = results(dev_gold, spacy_stan_lab)

    table.to_csv('./results/'+ stan_spacy + '_pre_rec_f1_LR.conll', sep='\t')
    confm.to_csv('./results/'+ stan_spacy + '_confm_LR.conll', sep='\t')

   
if __name__ == '__main__':
   main()

#type this on the terminal: 
#python3 basic_system.py '../../data/preprocessed/conll2003.train-preprocessed.conll' '../../data/preprocessed/conll2003.dev-preprocessed.conll' '../../models/output/output_conll.conll' 'conll'

