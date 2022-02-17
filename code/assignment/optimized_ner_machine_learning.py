from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
import gensim
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import sys
import csv

# Train models with features represented as one-hot encodings

feature_to_index = {'token': 0, 'pos': 1, 'chunk': 2, 'previous': 4, 'following': 5, 'capital': 6}

def extract_features_and_labels(trainingfile, selected_features):
    '''
    Extract features and gold labels from a preprocessed file with the training data and return them as lists
    
    :param trainingfile: path to training file
    :param selected_features: list of features that will be used to train the model
    
    :type trainingfile: string
    :type selected_features: list
    
    :return features: features as a list of dictionaries
    :return gold_labels: list of gold labels
    '''
    features = []
    gold_labels = []
    
    conllinput = open(trainingfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    
    for row in csvreader:
        feature_value = {}
        # Only extract the selected features
        for feature_name in selected_features:
            row_index = feature_to_index.get(feature_name)
            feature_value[feature_name] = row[row_index]
        features.append(feature_value)
        
        # Gold is in the third column
        gold_labels.append(row[3])
                
    return features, gold_labels
    
def extract_features(testfile, selected_features):
    '''Extract features from a preprocessed file with the test data and return them as a list
    
    :param trainingfile: path to test file
    :param selected_features: list of features that were selected to train the model
    
    :type testfile: string
    :type selected_features: list
    
    :return features: features as a list of dictionaries'''

    features = []
    gold_labels = []
    
    conllinput = open(testfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    
    for row in csvreader:
        feature_value = {}
        for feature_name in selected_features:
            row_index = feature_to_index.get(feature_name)
            feature_value[feature_name] = row[row_index]
        features.append(feature_value)
                
    return features
    
def create_classifier(train_features, train_labels, modelname):
    '''Create a classifier and train it with vectorized features and corresponding gold labels
    
    input train_features: features to be transformed into vectors
    input train_labels: gold labels corresponding to features
    input modelname: name of the model that will be trained
    
    output model: trained classifier
    output vec: DictVectorizer'''
    
    if modelname ==  'logreg':
        model = LogisticRegression(max_iter=10000)
    elif modelname == 'NB':
        model = MultinomialNB()
    elif modelname == 'SVM':
        model = svm.LinearSVC(max_iter=10000)
        
    vec = DictVectorizer()
    
    features_vectorized = vec.fit_transform(train_features)
    model.fit(features_vectorized, train_labels)
    
    return model, vec
    
def classify_data(model, vec, inputdata, outputfile, selected_features):
    '''Make predictions on test data and write the results to a file
    
    input model: classifier that will make predictions
    input vec: DictVectorizer object to transform the features into vectors
    input inputdata: path to input data
    input outputfile: path to output file where the predictions for each feature will be written
    input selected_features: list of features that were selected to train the model
    
    output: external file with results'''
    
    # Extract features from inputdata
    features = extract_features(inputdata, selected_features)
    features = vec.transform(features)
    
    # Make predictions
    predictions = model.predict(features)
    
    # Write results to an outputfile
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()
    
# Train model with only word embeddings as features
    
def extract_embeddings_as_features_and_gold(trainingfile, word_embedding_model):
    '''Extracts features and gold labels from file with training data using word embeddings
    
    input trainingfile: path to training file
    input word_embedding_model: a pretrained word embedding model
    
    output features: list of vector representation of tokens
    output gold_labels: list of gold labels'''

    features = []
    gold_labels = []
    

    conllinput = open(trainingfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    
    for row in csvreader:
        if row[0] in word_embedding_model:
            vector = word_embedding_model[row[0]]
        else:
            vector = [0]*300
        features.append(vector)
        
        # Gold is in the third column
        gold_labels.append(row[3])
        
    return features, gold_labels

def extract_embeddings_as_features(testfile, word_embedding_model):
    '''Extracts features from file with test data using word embeddings
    
    input testfile: path to conll file
    input word_embedding_model: a pretrained word embedding model
    
    output features: list of vector representation of tokens'''

    features = []


    conllinput = open(testfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    
    for row in csvreader:
        if row[0] in word_embedding_model:
            vector = word_embedding_model[row[0]]
        else:
            vector = [0]*300
        features.append(vector)

    return features

def create_classifier_embeddings(train_features, train_labels):
    '''Create an SVM classifier and train it with vectorized features and corresponding gold labels
    
    input train_features: features to be transformed into vectors
    input train_labels: gold labels corresponding to features
    
    output model: trained classifier'''

    model = svm.LinearSVC(max_iter=10000)
    model.fit(train_features, train_labels)
    
    return model
        
def classify_data_embeddings(model, inputdata, outputfile, word_embedding_model):
    '''Let a classifier make predictions on new data
    
    input model: classifier that will make predictions
    input inputdata: path to input data
    input outputfile: path to output file, where the predictions for each feature will be written
    input word_embed'''

    # Extract features from inputdata
    features = extract_embeddings_as_features(inputdata, word_embedding_model)
    
    # Make predictions
    predictions = model.predict(features)
    
    # Write results to an outputfile
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()
    

# Train mixed model (with embeddings and features represented as one-hot)

def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''Function that takes sparse and dense feature representations and appends their vector representation
    
    input dense_vectors: list of dense vector representations
    input sparse_features: list of sparse vector representations
    
    output: list of arrays in which sparse and dense vectors are concatenated '''
    
    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())
    
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
        
    return combined_vectors

def extract_traditional_features_and_embeddings_plus_gold_labels(conllfile, word_embedding_model, vectorizer=None):
    '''Function that extracts traditional features as well as embeddings and gold labels using word embeddings for current and preceding token
    
    input conllfile: path to conll file
    input word_embedding_model: a pretrained word embedding model
    
    output features: list of vector representation of tokens
    output labels: list of gold labels'''
    
    # Get the word embeddings and a selection of traditional features plus gold labels using the functions from before
    dense_vectors = extract_embeddings_as_features(conllfile, word_embedding_model)
    traditional_features,labels = extract_features_and_labels(conllfile, ['pos','chunk','capital'])
            
    # Create vector representation of traditional features
    if vectorizer is None:
        vectorizer = DictVectorizer()
        vectorizer.fit(traditional_features)
    sparse_features = vectorizer.transform(traditional_features)
    combined_vectors = combine_sparse_and_dense_features(dense_vectors, sparse_features)
    
    return combined_vectors, vectorizer, labels

def label_data_with_combined_features(testfile, outputfile, classifier, vectorizer, word_embedding_model):
    '''Labels data with model using both sparse and dense features'''
    
    # Extract features and gold labels from inputdata
    feature_vectors, vectorizer, goldlabels = extract_traditional_features_and_embeddings_plus_gold_labels(testfile, word_embedding_model,vectorizer)
    
    # Make predictions   
    predictions = classifier.predict(feature_vectors)
    
    # Write the results to an output file
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(testfile, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()
    
    
def main():
    
    argv = sys.argv
    
    trainingfile = argv[1] #path to training file
    inputfile = argv[2] #path to dev file that will be used to generate output
    outputfile = argv[3] # name of output file that will be created
    type_of_system = argv[4] # 0 == traditional, 1==embedding, 2==mixed
    language_model = '../../models/GoogleNews-vectors-negative300.bin.gz' #if at some point there are others, add as argument argv[5]

    
    if type_of_system == '0': #traditional_system
        
        # Select features to train the model (update list for feature ablation)
        selected_features = ['token', 'pos', 'chunk', 'previous', 'following', 'capital']
    
        # Get the selected training features and gold labels
        training_features, gold_labels = extract_features_and_labels(trainingfile, selected_features)

        # Train three different models with the features, let them classify the data and write the result to new conll files
        for modelname in ['logreg', 'NB', 'SVM']:
            ml_model, vec = create_classifier(training_features, gold_labels, modelname)
            classify_data(ml_model, vec, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'), selected_features)
            
            df = pd.read_table(outputfile.replace('.conll','.' + modelname + '.conll'))
            df = df.set_axis([*df.columns[:-1], 'NER2'], axis=1, inplace=False)
            df.to_csv(outputfile.replace('.conll','.' + modelname + '.conll'), sep='\t')
            
    elif type_of_system == '1': #embeddings_system
        
        # Load word embedding model
        language_model = gensim.models.KeyedVectors.load_word2vec_format(language_model, binary=True)
        
        # Get training features and gold labels
        training_features, gold_labels = extract_embeddings_as_features_and_gold(trainingfile, language_model)
        
        # Train an SVM classifier with the features, let it classify the data and write result to conll file
        ml_model = create_classifier_embeddings(training_features, gold_labels)
        classify_data_embeddings(ml_model, inputfile, outputfile.replace('.conll','.embed.conll'), language_model)
        
        df = pd.read_table(outputfile.replace('.conll','.embed.conll'))
        df = df.set_axis([*df.columns[:-1], 'NER2'], axis=1, inplace=False)
        df.to_csv(outputfile.replace('.conll','.embed.conll'), sep='\t')
        
    elif type_of_system == '2': #mixed_system
        
        # Load word embedding model
        language_model = gensim.models.KeyedVectors.load_word2vec_format(language_model, binary=True)
        
        # Get the combined feature vectors for training, vectorizer and gold labels
        combined_vectors, vectorizer, gold_labels = extract_traditional_features_and_embeddings_plus_gold_labels(inputfile, language_model, vectorizer=None)
        
        # Train an SVM classifier with the features, let it classify the data and write result to conll file
        ml_model = create_classifier_embeddings(combined_vectors, gold_labels)
        label_data_with_combined_features(inputfile,outputfile.replace('.conll','.mixed.conll'),ml_model,vectorizer,language_model)

        df = pd.read_table(outputfile.replace('.conll','.mixed.conll'))
        df = df.set_axis([*df.columns[:-1], 'NER2'], axis=1, inplace=False)
        df.to_csv(outputfile.replace('.conll','.mixed.conll'), sep='\t')
        
    
if __name__ == '__main__':
    main()


#python3 optimized_ner_machine_learning.py ../../data/preprocessed/conll2003.train-preprocessed_with_features.conll ../../data/preprocessed/conll2003.dev-preprocessed_with_features.conll ../../models/output/output_system.conll 0
#python3 optimized_ner_machine_learning.py ../../data/preprocessed/conll2003.train-preprocessed_with_features.conll ../../data/preprocessed/conll2003.dev-preprocessed_with_features.conll ../../models/output/output_system.conll 1
#python3 optimized_ner_machine_learning.py ../../data/preprocessed/conll2003.train-preprocessed_with_features.conll ../../data/preprocessed/conll2003.dev-preprocessed_with_features.conll ../../models/output/output_system.conll 2

#for ablation tests we have 6 features in total. remove one by one from last until we only have first
#eg: remove the last one, AKA 'capital' has the name output_system_rm1.conll
#remove 'capital' and 'following' has the name output_system_rm2.conll