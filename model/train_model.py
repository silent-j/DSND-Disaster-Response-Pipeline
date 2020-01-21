# -*- coding: utf-8 -*-
"""
train_model.py [module]
@author: silent-j
Train and evaluate a machine learning pipeline for multi-label classification

Tune hyperparameters using GridSearch cross-validation and save selected
estimator to a pickle object

Args:
database_path (str): 
    path to a .db file, containing the text data
model_path (str):
    if train = True, model_path is used to serialize a trained model
    else, model_path is used to load an already trained model

"""
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, make_scorer, f1_score


def load_data(db_path, table_name):
    """
    Func: load data from an sql database using Pandas
    Parameters:
    db_path: str, path to the sql database
    table_name: str, table name to select data from
    Returns:
    X: dataframe of text data, (n_samples, )
    y: dataframe of class data, (n_samples, n_classes)
    labels: list of class labels
    """    
    engine = create_engine(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    X = df['message']
    y = df.iloc[:, 4:]
    labels = y.columns.values

    return X, y, labels

def tokenize(text):
    """
    Func: normalize, tokenize, and lemmatize inputted text using nltk
    Parameters:
    text: str
    Returns:
    tokens: list of tokens which have been processed
    """    
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    text = re.sub('[^a-zA-Z0-9\s]', '', text.lower())  # normalize  
    tokens = word_tokenize(text)  # tokenize  
    tokens = [word.strip() for word in tokens if not word in stop_words] # stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # lemmatize
    
    return tokens

def build_pipeline(X_train, y_train):
    """
    Func: build a machine pipeline for text classification and tune 
    hyperparameters using GridSearchCV
    Parameters:
    None
    Returns:
    Selected best_estimator_ from the trained GridSearch
    """
    pipeline = Pipeline([
                ('vect', TfidfVectorizer(tokenizer=tokenize)),
                ('lsa', TruncatedSVD(n_components=100, random_state=42)),
                ('clf', MultiOutputClassifier(MLPClassifier(random_state=42)))
    ])
    # GridSearch
    parameters = {'vect__ngram_range': [(1,1), (1,2)],
                'vect__max_df': [0.7, 1.0],
                'vect__max_features': [None, 5000],
                'lsa__n_components': [50, 100, 200],
                'clf__estimator__learning_rate_init': [0.001, 0.0001],
                'clf__estimator__max_iter': [500],
                'clf__estimator__n_iter_no_change': [5],
                'clf__estimator__warm_start': [True],
                'clf__estimator__early_stopping': [True],
                'clf__estimator__random_state': [42]}
    cv = GridSearchCV(pipeline, param_grid=parameters, 
                      scoring=make_scorer(f1_score, average='weighted'),
                      n_jobs=4, cv=5, verbose=10)
    
    cv.fit(X_train, y_train)
    # select best estimator
    model = cv.best_estimator_
   
    return model

def save_model(path, model):
    """
    Func: save model to pickle object
    Parameters:
    path: str, path to save pickle object to
    model: trained instance of a model class
    Returns:
    None
    """
    with open(path, "wb") as out:
        pickle.dump(model, out)
    
def load_model(path):
    """
    Func: load a saved instance of a model class using joblib
    Parameters:
    path: str, path to saved pickle object
    Returns:
    saved instance of a model class
    """
    model = joblib.load(path)
    
    return model

def evaluate(y_true, y_pred):
    """
    Func: evaluate the performance of a classifier model. Print out global & 
    label evaluation metrics.
    Parameters:
    y_true: array of ground-truth labels, (n_samples, n_labels)
    y_pred: array of predicted labels, (n_samples, n_labels)
    Returns:
    Pandas dataframe of label evaluation metrics: precision, recall, 
    Unweighted F-Score and support
    """
    result = precision_recall_fscore_support(y_true, y_pred)
    scores = []
    for i, col in enumerate(y_true.columns.values):
        scores.append((result[3][i], result[0][i], result[1][i], result[2][i]))
    
    score_df = pd.DataFrame(index=y_true.columns.values, data=scores, 
                            columns=['Total Positive labels', 'Precision', 
                                     'Recall', 'Unweighted F-Score'])
    score_df.sort_values(by='Unweighted F-Score', axis=0, 
                         ascending=False, inplace=True)

    acc = accuracy_score(y_true, y_pred)
    loss = hamming_loss(y_true, y_pred)
    print("=====Global Metrics=====\n")
    print("Accuracy: {:.4f}".format(acc))
    print("Hamming Loss: {:.4f}\n".format(loss))
    print("=====Label Metrics=====\n")
    
    return score_df

def main():
    
    database_path, model_path = sys.argv[1:]
    
    X, y, labels = load_data(database_path, "DisasterTab")
    
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)
    
    # train & tune model
    model = build_pipeline(X_train, y_train)
    # test & evaluate    
    predictions = model.predict(X_test)
    # print out label metrics
    score_df = evaluate(y_test, predictions)
    print(score_df)

