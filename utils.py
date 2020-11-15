import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def load_dataset(filepath):
    '''
    Doc...
    '''
    obj = []
    with open(filepath, 'r', encoding="UTF-8") as file:
        obj = json.load(file)
    return obj


def prepare_X_y(train, test):
    '''
    doc...
    '''
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for doc in train:
        X_train.append(doc['question'])
        y_train.append(doc['category'])
        
    for doc in test:
        X_test.append(doc['question'])
        y_test.append(doc['category'])

    
    return X_train, y_train, X_test, y_test


def extract_features(train_dataset, test_dataset, file):
    """Extracts feature vectors from a preprocessed train and test datasets.
    
    Args:
        train_dataset: List of strings, each consisting of the preprocessed email content. 
        test_dataset: List of strings, each consisting of the preprocessed email content. 
    
    Returns: train and test vectors
        
    """
    vectorizer = CountVectorizer()
    train_vectors = vectorizer.fit_transform(train_dataset)
    with open(file, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    test_vectors = vectorizer.transform(test_dataset)
    return train_vectors, test_vectors


def transform_dataset(test_dataset, file):
    with open(file, 'rb') as f:
        return pickle.load(f).transform(test_dataset)
    return None


def split_bool_literal_reference(X, y):
    """
    doc...
    """
    bool_map = {}
    literal_map = {}
    resource_map = {}

    for i in range(len(y)):
        if y[i] == 'boolean':
            bool_map[i] = X[i]
        elif y[i] == 'literal':
            literal_map[i] = X[i]
        elif y[i] == 'resource':
            resource_map[i] = X[i]
            
    return bool_map, literal_map, resource_map

