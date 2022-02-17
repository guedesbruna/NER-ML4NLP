# ma-ml4nlp-labs

## Repository overview

This repository provides scripts for the master course 'Machine Learning in NLP' offered by VU Amsterdam.

This repository is organized as follows:
``` bash
/ML4NLP
├── README.md
├── LICENSE
├── requirements.txt
├── data
│   └── preprocessed
├── code
│   ├── assignment
│   │   ├── results
│   │   ├── preprocessing_conll_2021.py
│   │   ├── add_features.py
│   │   ├── basic_evaluation.py
│   │   ├── basic_system.py
│   │   ├── optimized_ner_machine_learning.py
│   │   ├── bert4ner
│   │   │   ├── bert_finetuner.ipynb
│   │   │   ├── bert_utils.py
│   │   │   ├── requirements2.txt
│   │   │   ├── README2.md
│   │   │   ├── data
│   │   │   └── saved_models
│   └── settings
│   │   └── conversions.tsv
├── models
│   └── output
```

## Running the code - STEPS
1. make sure to have all packages specified in requirement.txt
2. add model GoogleNews-vectors-negative300.bin.gz under /models folder
3. add raw datafiles under /data folder. For the purpose of this task you shoudl add 4 files:
    - conll2003.train.conll, conll2003.dev.conll, spacy_out.dev.conll, stanford_out.dev.conll
4. change directory until positioned in /code/assignment folder
5. run preprocessing_conll_2021.py by typing on the terminal:
    - python3 preprocessing_conll_2021.py
        - this will generate 7 files: 3 files that end in preprocessed0.conll under the /data folder and 4 files that end in preprocessed.conll under the folder /data/preprocessed.
6. run add_features.py by running the below line on the terminal:
    - python3 add_features.py '../../data/preprocessed/conll2003.train-preprocessed.conll' '../../data/preprocessed/conll2003.dev-preprocessed.conll'
        - this will generate 2 files that will contain new features. The files names will finish in with_features.conll and will be under the folder ../../data/preprocessed.
7. run basic_system.py by typing the following line on the terminal:
    - python3 basic_system.py '../../data/preprocessed/conll2003.train-preprocessed.conll' '../../data/preprocessed/conll2003.dev-preprocessed.conll' '../../models/output/output_conll.conll' 'conll'
        - this will generate the output with the prediction for a basic system in ../../models/output (1 output)
        - it will generate the confusion matrix and precision_recall_f1 table in ./results (2 outputs)
8. run optimized_ner_machine_learning.py by typing the following line on the terminal to calculate output from the preprocessed file with additional features:
    - for traditional models (LR, NB, SVM): python3 optimized_ner_machine_learning.py ../../data/preprocessed/conll2003.train-preprocessed_with_features.conll ../../data/preprocessed/conll2003.dev-preprocessed_with_features.conll ../../models/output/output_system.conll 0
        - this will generate 3 outputs, one for each model (indicated by the name of the output file), under the folder ../../models/output
    - for embedding models: python3 optimized_ner_machine_learning.py ../../data/preprocessed/conll2003.train-preprocessed_with_features.conll ../../data/preprocessed/conll2003.dev-preprocessed_with_features.conll ../../models/output/output_system.conll 1
        - this will generate 1 output, indicated wit the 'embed' termination in the name of the output file, under the folder ../../models/output
    - for mixed models: python3 optimized_ner_machine_learning.py ../../data/preprocessed/conll2003.train-preprocessed_with_features.conll ../../data/preprocessed/conll2003.dev-preprocessed_with_features.conll ../../models/output/output_system.conll 2
        - this will generate 1 output, indicated wit the 'mixed' termination in the name of the output file, under the folder ../../models/output
    - for ablation, change inside the scrip the variable  selected_features including only the features that should be taken into account by the model. Save the script and run the lines above.
9. run basic_evaluation.py to evaluate the outputs of the models. The following line of code is to evaluate the output of SVM code: 
    - python3 basic_evaluation.py '../../data/preprocessed/conll2003.train-preprocessed.conll' '../../models/output/output_system.SVM.conll' conll_SVM
    - this generates 2 files under the folder ./result: conll_SVM_confm.conll (for the confusion matrix) and conll_SVM_prec_rec_f1.conll (for the precision, recall and f1)
    - this command line should be repeated for all models that one wishes to see the metrics. The order is: 
        - python3 basic_evaluation.py path_training_data path_system_output name_of_metric_file
10. for running BERT, please position the following files to /code/assignment/bert4ner/data:
    - conll2003.train.conll, conll2003.dev.conll, conll2003.test.conll
11. Open notebook bert_finetuner.ipynb that can be found under /code/assignment/bert4ner run it as it is and see the results inside the notebook itself.
    - Disclaimer: to run this notebooks additional packages should be installed. They will be installed automatically on the first cell if the cell is not commented.
 
 **Disclaimer**
 
 The code from this repository was based on the repository from: https://github.com/cltl/ma-ml4nlp-labs 
