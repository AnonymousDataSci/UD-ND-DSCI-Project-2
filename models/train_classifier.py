import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import numpy as np
import pandas as pd

import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

def load_data(database_filepath):
    """
    Load Data into the Pipeline
    Arguments:
    database filepath: Filepath to sql data
   
    Return:
    x: dataframe  features
    y: dataframe labels
    catergory names: list of categories
    """
    #Load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    
    #Sperate into x and y 
    df = pd.read_sql_table('DisasterMessages',engine)
    X = df['message']
    y = df.iloc[:,4:]
    
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """
    Tokenize Data in the Pipeline
    Arguments:
    text: text messages from figure 8
   
    Return:
    clean token: cleaned tokens from the text mesage
    """
    #Replace all urls with a placeholder and extrat all urls from the text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    #Extract the token 
    tokens = word_tokenize(text)
    
    #Lematize the tokens
    lemmatizer = WordNetLemmatizer()

    #List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Build the model of the pipeline
   
    Return:
    pipeline: return the pipline of the model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', RandomForestClassifier())
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model the Pipeline
    Arguments:
    model: the model build with scikit
    x test: features
    y test: labels
    category names: catergory names

    """
    y_pred = model.predict(X_test)
    
    # Evaluate the model on the test data
    for i in range(len(category_names)):
        print("Category: ", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:,i])))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()