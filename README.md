# room-locator

Room-locator is decision tree based machine learning program that determines the location of signal receptor out of 4 rooms. It takes in a dataset of a large number of samples each one containing 7 wifi signal strength measures and the actual location of the receptor. It performs a cross-validation evaluation that splits the dataset into 10 folds and iteratively computes a decision tree for each fold and outputs averaged metrics (accuracy, precision, recall and F1 measure) of our program's predictions on the overall dataset. 

# installation

Clone our GitHub repository on your prefered IDE.
Make sure to have the python 'numpy' and 'matplotlib' libraries installed.

# usage

Our program has two main functionalities, both ran from the 'main.py' file:
    - 'generate_report()' will generate a report (in .txt) outputing the decision tree and analytics of the 'clean_dataset.txt' and 'noisy_dataset.txt' datasets. 
    - 'predict()' will allow users to upload their own datasets and will predict the label of their own samples. 