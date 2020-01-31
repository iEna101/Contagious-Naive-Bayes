#Functions:
'''
1.   preprocessing - Simple pre-processing to remove noise from corpus
2.   features - Obtain feature (document x word) matrix for a set of input documents
3.   class_interest - Extract class specific feature matrix
4.   class_probability - Compute class prior
5.   likelihood_cnb - Computes the MLE's for the contagious naive Bayes classifier
6.   posterior_cnb - Computes the posterior probabilities associated with the test scenarios
7.  CNB_classification - Computes standard evaluation metrics such as accuracy, precision, recall and f1;
'''

import pandas as pd
import numpy as np

import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import warnings
warnings.filterwarnings(action = 'ignore')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from bs4 import BeautifulSoup

def preprocessing(file):
    '''
    Input:
        File: Text documents
    Output:
        Lemmatized_out: List filled with preprocessed text
    Description:
        Simple pre-processing to remove noise from corpus
    '''
    lemmatizer = WordNetLemmatizer()
    for row in file:
        sentence = str(row)
        # Set all characters to lowercase
        lower_case = sentence.lower()
        # Remove non-alphabetic characters
        no_alpha = re.sub(r'[^\w\s]','',lower_case)
        # Remove underscore character
        no_under = re.sub(r"_","",no_alpha)
        # Tokenize
        token = word_tokenize(no_under)
        # Lemmatize
        lemmatized_out = ' '.join([lemmatizer.lemmatize(w) for w in token])

        return lemmatized_out
    
def features(file):
    '''
    Input:
        File: Text documents
    Output:
        Feature_mat: Bag of words (BoW) feature matrix
    Description:
        Obtain feature (document x word) matrix for a set of input documents
    '''
    matrix = CountVectorizer(stop_words='english')
    feature_mat = matrix.fit_transform(file).toarray()
    feature_mat = pd.DataFrame(feature_mat)
    feature_names = matrix.get_feature_names()
    feature_mat.columns = feature_names
    
    return feature_mat

def class_interest(file, class_id):
    '''
    Input:
        File: Bag of words feature matrix
        class_id: Integer that represents class of interest in class column
                    0: Negative
                    1: Positive
    Output:
        class_f: Bag of words (BoW) for class of interest
    Description:
        Extract class specific feature matrix
    '''
    class_f = file.loc[file['class_code'] == class_id]
    class_f = class_f.reset_index()
    class_f = class_f.drop(['class_code'], axis = 1)
    
    return class_f

def class_probability(file,class_id):
    '''
    Input:
        File: Text documents
        class_id: Integer that represents class of interest in class column
                    0: Negative
                    1: Positive
    Output:
        Class_prob: Probability of class of interest 
    Description:
        Compute class prior
    '''
    class_file = class_interest(file,class_id)
    class_count = len(class_file)
    total_count = len(file)
    class_prob = np.true_divide(class_count,total_count)
    
    return class_prob

def likelihood_cnb(file,class_id,c1,c2):
    '''
    Input:
        file: Feature matrix of Text documents
        class_id: Integer that represents class of interest in class column
                    0: Negative
                    1: Positive
        c1: Smoothing parameter - numerator
        c2: Smoothing parameter - denominator
    output:
        likelihood: MLE's based on training scenarios
    description:
        Computes the MLE's for the contagious naive Bayes classifier
    '''
    class_file = class_interest(file,class_id)
    class_file = class_file.drop(['id_labels'],axis=1)
    feature_names = class_file.columns
    D_c = len(class_file)
    sum_x_ij = []
    for n in class_file[feature_names]:
        sum_n = sum(class_file[n])
        sum_x_ij.append(sum_n)
        n_k = list(sum_x_ij)
        #print(sum_x_ij,n_k)
        denominator = np.add(D_c,c2)
        numerator = np.add(n_k,c1)
        fraction = np.true_divide(numerator,denominator)
        
    theta_hat = pd.DataFrame(fraction).transpose()
    theta_hat.columns = feature_names
    
    return(theta_hat)

def posterior_cnb(test_file,train_features,class_id,c1,c2):
    '''
    input:
        test_file: Test case
        train_features: Feature matrix of text document (BoW)
        class_id: Integer that represents class of interest in class column
                    0: Negative
                    1: Positive
        c1: Smoothing parameter - numerator
        c2: Smoothing parameter - denominator
    output:
        posterior: Posterior probability of test scenarios for class of interest
    description:
        Computes the posterior probabilities associated with the test scenarios
    '''
    c_hat = []
    test_features = features(test_file) # features present in test case
    train_cnb = likelihood_cnb(train_features,class_id,c1,c2)
    for j in range(len(test_features)):
        test_j_feature = test_features.loc[j] # doc_j in test set
        test_j_feature = test_j_feature.iloc[test_j_feature.nonzero()[0]] # features present in test case
        # class prior
        class_p = class_probability(train_features,class_id)
        class_p_log_space = np.log10(class_p)
        # conditional
        test_intersection = set(train_cnb.columns).intersection(test_j_feature.index)
        class_mle = train_cnb[test_intersection]
        conditional_log_space = (np.log10(class_mle) - class_mle)
        # Decision rule
        c_hat_j = np.add(class_p_log_space,np.sum(conditional_log_space,axis=1))
        c_hat.append(c_hat_j)
        
    return(c_hat)
def CNB_classification(feature_matrix,test_doc_clean,test_matrix,c1,c2):
    '''
    input:
        feature_matrix: Feature matrix of text document (BoW)
        test_doc_clean: Pre-processed test scenarios
        c1: Smoothing parameter - numerator
        c2: Smoothing parameter - denominator
    output:
        metric_frame: Standard ckassification metrics
    description:
        Computes standard evaluation metrics such as accuracy, precision, recall and f1
    '''
    class_code = np.unique(feature_matrix['class_code'])
    f = []
    for i in class_code:
        c_hat_test = posterior_cnb(test_doc_clean,feature_matrix,i,c1,c2)
        for j in range(len(c_hat_test)):
            c_frame = (c_hat_test[j].values)
            frame = (i,j,c_frame[0])
            frame = list(frame)
            f.append(frame)
    
    c_hat_class = pd.DataFrame(f,columns = ['class_code','j','c_hat'])
    c_hat_doc = c_hat_class.groupby(['j']).agg(lambda x: list(x))
    c_hat_doc = c_hat_doc.reset_index()
    decision_doc_cnb = c_hat_doc['c_hat']
    decision_doc_cnb = pd.DataFrame(list(decision_doc_cnb))
    decision_doc_cnb['Index'] = test_doc_clean.index
    
    decision_doc_cnb['Predicted'] = (decision_doc_cnb[0] < decision_doc_cnb[1]).astype(int) # will return 0 if [0] less than [1]
    decision_doc_cnb = decision_doc_cnb.set_index(decision_doc_cnb['Index'])
    decision_doc_cnb['Actual'] = test_matrix['class_code'].astype(int)

    #return decision_doc_cnb

    # Classification metrics
    acc_cnb = accuracy_score(decision_doc_cnb['Actual'], decision_doc_cnb['Predicted'])
    precision_cnb = precision_score(decision_doc_cnb['Actual'], decision_doc_cnb['Predicted'])
    recall_cnb = recall_score(decision_doc_cnb['Actual'], decision_doc_cnb['Predicted'])
    f1_cnb = f1_score(decision_doc_cnb['Actual'], decision_doc_cnb['Predicted'])

    metric_frame = (acc_cnb,precision_cnb,recall_cnb,f1_cnb)

    return (decision_doc_cnb,metric_frame)

''''''''''''''''''''''''''''''''''''''''''''''''

def CNB(train_matrix,test_matrix, norm = False, pseduo_len = 100, c1 = 0.001, c2 = 1):
  if pseduo_len <= 0:
    print("\033[1;31;47m Please make sure your Pseduo length is positive.  \n")
    return   
    
  import time
  t0 = time.time()

  if norm is False:
    data = [str(i) for i in train_matrix['text']]

    feature_matrix = features(data)
    train_matrix_id = train_matrix.index
    feature_matrix['id_labels'] = train_matrix_id
    feature_matrix = feature_matrix.set_index(feature_matrix['id_labels'])
    feature_matrix = feature_matrix.drop(columns = ['id_labels'])
    
    feature_matrix["class_code"] = train_matrix["class_code"] # Add class column to feature matrix to seperate classes
    test_matrix.columns = ['text','class_code']
    test_doc = pd.DataFrame(test_matrix)
    test_doc_clean = test_doc.apply(lambda x: preprocessing(x), axis = 1)
  
  else:
    corpus_clean = train_matrix.apply(lambda x: preprocessing(x), axis = 1)
    
    doc_new = pd.DataFrame(corpus_clean,columns = ['text'])
    doc_new['class_code'] = train_matrix['class_code'].astype(int)
    doc_new_id = doc_new.index
    
    doc_nc = doc_new.groupby(['class_code']).agg(lambda x: list(x))
    doc_nc = doc_nc.reset_index()
    cols = ['text','class_code']
    doc_nc = doc_nc[cols]
    doc_new_c = pd.DataFrame(doc_nc.apply(lambda x: preprocessing(x),axis=1),columns = ['text'])
    doc_new_c['class_code'] = doc_nc['class_code']
    
    class_codes = np.unique(doc_new_c['class_code'])
    df = pd.DataFrame(columns = ['class_code','text'])
    for i in class_codes:
        ci = class_interest(doc_new_c,i)
        for n in ci['text']:
            doc_len = len(n.split())
            pseudo_len = 124
            pseudo = round(doc_len / pseudo_len)
            pseudo_doc = [n.split()[k:k + pseudo_len] for k in range(0,doc_len,pseudo_len)]
            pseudo_doc = [' '.join(pseudo_doc[k]) for k in range(0,len(pseudo_doc))]
           
            pseudo_doc_class = ([i]*len(pseudo_doc))
            for j in range(0,len(pseudo_doc)):
              pseudo_jj = (pseudo_doc_class[j], pseudo_doc[j])
              pseudo_jj = list(pseudo_jj)
              pseudo_df = pd.Series(pseudo_jj, index = ["class_code","text"])
              df = df.append(pseudo_df, ignore_index = True)
                
    # Feature matrix for DLN data
    feature_matrix = features(df['text'])
    
    feature_matrix["class_code"] = df["class_code"] # Add class column to feature matrix to seperate classes
    feature_matrix_ids = feature_matrix.index
    feature_matrix['id_labels'] = feature_matrix_ids
    test_doc = pd.DataFrame(test_matrix)
    test_doc_clean = test_doc.apply(lambda x: preprocessing(x), axis = 1)

    #Contagious Naive Bayes
  x_cnb = CNB_classification(feature_matrix,test_doc_clean,test_matrix,c1,c2)
  print('The Contagious Naive Bayes has executed.')

  t1 = time.time()
  total = t1 - t0
  print("The total runtime was: ",total, "seconds")
    
  Post_Probs, Metrics = x_cnb
  names = ["Accuracy", "Precision", "Recall", "F1"]

  Results = pd.DataFrame(Metrics, names)

  print("The posteriors obtained are as follows: ")
  Post_Probs = Post_Probs.drop(Post_Probs.columns[2], axis=1)
  print(Post_Probs)
  print('The performance metrics obtained are as follows: ', Results)
  return(Post_Probs, Metrics)
