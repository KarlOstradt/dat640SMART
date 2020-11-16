# DAT640 SMART

This repository comprises of multiple jupyter notebooks. Each notebook is used to fullfill a single task. Here is an overview of what each file is used for.

> **Split_Dataset.ipynb**  
> Split the original dataset into train and test set.  
> Outputs the files: 
> - datasets/DBpedia/train.json
> - datasets/DBpedia/test.json
> - datasets/DBpedia/test_grnd.json

> **Indexing.ipynb**  
> Run this notebook to index the training set using Elasticsearch.

> **Word2Vec.ipynb**  
> Trains the word2vec model.
> Outputs the files: 
> - word2vec_sg.sav

> **Category_classifier.ipynb**  
> Trains both the baseline and advanced category classifiers. 
> Outputs the files:  
> - category_classifier_baseline.sav
> - category_classifier_advanced.sav
> - category_vectorizer.sav

> **Literal_classifier.ipynb**  
> Trains literal type classifier. 
> Outputs the files:  
> - literal_vectorizer.sav
> - type_literal_classifier.sav

> **utils.py**  
> Utility methods used accross multiple notebooks.

> **SMART_baseline.ipynb**  
> The baseline method, predicting the category and answer types on the test set. Requires that **Indexing.ipynb** has been run. The results are printed at the end in this notebook.  
> Outputs files:  
> - datasets/DBpedia/pred_baseline.json

> **SMART_advanced.ipynb**  
> The advanced method, predicting the category and answer types on the test set. Requires that **Indexing.ipynb** has been run. The results are printed at the end in this notebook.  
> Outputs files:  
> - datasets/DBpedia/pred_advanced.json

> **datasets/DBpedia/smarttast_dbpedia_train.json**  
> The original dataset

> **evaluation/dbpedia/evaluation.py**  
> The evaluation script

> **evaluation/dbpedia/dbpedia_types.tsv**  
> The DBpedia ontology hierarchy.


Contributors: Karl Østrådt and Eirik Haraldsen