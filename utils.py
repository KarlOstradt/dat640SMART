import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def load_dataset(filepath):
    """ Load a dataset from a given filepath.
    
    Arguments:
        filepath: Path to dataset file in json format.
    """
    obj = []
    with open(filepath, 'r', encoding="UTF-8") as file:
        obj = json.load(file)
    return obj


def prepare_X_y(train, test):
    """ Prepare datasets for feature extraction.
    
    Arguments:
        train: Dataset used for training.
        test: Dataset used for testing.
        
    Returns:
        X_train: List of all questions in training dataset.
        y_train: List of all category labels in training dataset.
        X_test: List of all questions in test dataset.
        y_test: List of all category labels in test dataset.
    """
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
    """Extracts feature vectors from train and test datasets.
    
    Args:
        train_dataset: List of strings, each consiting of all training questions 
        test_dataset: List of strings, each consiting of all test questions  
    
    Returns: 
        train_vectors: frequency feature vectors for training
        test vectors: frequency feature vectors for test
        
    """
    vectorizer = CountVectorizer()
    train_vectors = vectorizer.fit_transform(train_dataset)
    with open(file, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    test_vectors = vectorizer.transform(test_dataset)
    return train_vectors, test_vectors


def transform_dataset(dataset, file):
    """ Extract feature vectors from dataset using vectorizer.
    
    Arguments:
        dataset: Dataset to extract feature vector from.
        
    Returns: Feature vector or None
    """
    with open(file, 'rb') as f:
        return pickle.load(f).transform(dataset)
    return None


def split_bool_literal_reference(dataset, labels):
    """ Seperate dataset into boolean, literal and resource map.
        Each map uses index in dataset as key and question entity as value.
    
    Arguments:
        dataset: dataset containing question entities.
        labels: predicted answer category labels.
        
    Returns:
        bool_map: Map with predicted boolean questions.
        literal_map: Map with predicted literal questions.
        resource_map: Map with predicted resource questions.
    """
    bool_map = {}
    literal_map = {}
    resource_map = {}

    for i in range(len(labels)):
        if labels[i] == 'boolean':
            bool_map[i] = dataset[i]
        elif labels[i] == 'literal':
            literal_map[i] = dataset[i]
        elif labels[i] == 'resource':
            resource_map[i] = dataset[i]
            
    return bool_map, literal_map, resource_map

