import sys
import pickle
import re
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(database_filepath):
    '''
    input: database_filepath
    output: X: features dataframe, y: target dataframe
        category_names: names of targets
    '''
    # table name
    table_name = 'disaster_data'
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, con=engine)

    category_names = df.columns[4:]
    df = df.dropna() 
    X = df[['message']].values[:, 0]
    y = df[category_names].values

    return X, y, category_names
    
def tokenize(text):
    # stopword list 
    STOPWORDS = list(set(stopwords.words('english')))
    # initialize lemmatier
    lemmatizer = WordNetLemmatizer()
    # split string into words (tokens)
    tokens = word_tokenize(text)
    processed_tokens = []
    for token in tokens:
        # remove short words
        if len(token) > 2:
            # put words into base form
            lemmatizer.lemmatize(token).lower().strip()
            if token not in STOPWORDS:
                token = lemmatizer.lemmatize(token).lower().strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
                token = re.sub(r'\[[^.,;:]]*\]', '', token)
                if token != '':
                    processed_tokens.append(token)
    # return data 
    return processed_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=100)))
    ])
    # hyper-parameter grid
    param_grid = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__max_features': ['log2', 'auto'],
        'clf__estimator__n_estimators': [30],
    }
   
    # create model 
    cv = GridSearchCV(pipeline, param_grid=param_grid, verbose=2, n_jobs=4, cv=3)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    # print classification report
    for i in range(0, len(category_names)):
        print(category_names[i])
        print("\tAccuracy: {:.4f}\t\t% Precision: {:.4f}\t\t% Recall: {:.4f}\t\t% F1_score: {:.4f}".format(
            accuracy_score(Y_test[:, i], Y_pred[:, i]),
            precision_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            recall_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            f1_score(Y_test[:, i], Y_pred[:, i], average='weighted')
        ))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test == Y_pred)))

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