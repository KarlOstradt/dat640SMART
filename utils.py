import json
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


def extract_features(train_dataset, test_dataset):
    """Extracts feature vectors from a preprocessed train and test datasets.
    
    Args:
        train_dataset: List of strings, each consisting of the preprocessed email content. 
        test_dataset: List of strings, each consisting of the preprocessed email content. 
    
    Returns: train and test vectors
        
    """
    vectorizer = CountVectorizer()
    train_vectors = vectorizer.fit_transform(train_dataset)
    
    test_vectors = vectorizer.transform(test_dataset)
    return train_vectors, test_vectors