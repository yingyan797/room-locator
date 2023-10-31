# room-locator

Room-locator is decision tree based machine learning program that determines the location of signal receptor out of 4 rooms. It takes in a dataset of a large number of samples each one containing 7 wifi signal strength measures and the actual location of the receptor. It performs a cross-validation evaluation that splits the dataset into 10 folds and iteratively computes a decision tree for each fold and outputs averaged metrics (accuracy, precision, recall and F1 measure) of our program's predictions on the overall dataset. 

# installation

Clone our GitHub repository on your prefered IDE.
Make sure to have the required packages installed from the requirements.txt file.
```pip install -r requirements.txt```

# usage

To run the code and see results, run dashboard.py and follow the local host outputed in your terminal

Our program has two main functionalities, all accessible from the webpage:
- Train a model based on the clean/noisy datasets or your own uploaded dataset, and:
    - evaluate the performance of the model by performing a cross validation
    - create the decision tree and visualise it using matplotlib
- Make predictions based on inputed samples (manually or uploaded)