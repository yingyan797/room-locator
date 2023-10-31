# room-locator

Room-locator is decision tree based machine learning program that determines the location of signal receptor out of 4 rooms. It takes in a dataset of a large number of samples each one containing 7 wifi signal strength measures and the actual location of the receptor. It performs a cross-validation evaluation that splits the dataset into 10 folds and iteratively computes a decision tree for each fold and outputs averaged metrics (accuracy, precision, recall and F1 measure) of our program's predictions on the overall dataset. 

# installation

Before running the code make sure to have flask,numpy and matplotlib. In order for you to install them run the following commands in your terminal depending on the packages you need
```pip install flask```
```pip install numpy```
```pip install matplotlib```

# General usage

To run the code and see results, run dashboard.py and follow the local host  outputed in your terminal.
After copying the url in your web browser you can start running our program.

Our program has two main functionalities, all accessible from the webpage:
- Train a model based on the clean/noisy datasets or your own uploaded dataset, and:
    - Evaluate the performance of the model by performing a cross validation
    - Create the decision tree and visualise it using matplotlib
- Make predictions based on inputed samples (manually or uploaded)

# Training
- Before doing anything else, you need to train your model based on the provided dataset. For that, make sure to press the "Create decision tree" button before anything else (WARNING: If you don't do this it will not work correctly)
- If you want to select a different dataset, make sure to press the "Reselect dataset" button

# Visualization
- Simply push the "draw decision tree" button and a number of graphs will be created representing your decision tree
    - The tree will be drawn across multiple plots and divided into subtress in order for every node to be properly visualized.
    - If you're not able to properly see the information of each split point make sure to zoom in or download the image if you still have problems.

# Validation
- Just press the Cross validation button and it will show you both the confusion matrix and the relevant evaluation statistics:
       - the accuracy
       - the recall
       - the precision
       - the F1-measure

# Predict
- If you want to test the decision tree with new data you can either upload a file or manually input values for the different attributes:
    - If you upload a file make sure that it clears the following condition:
        - It's a .txt file
        - It's located inside the provided "predictions directory"
        - It has the same formatting as the provided clean and noisy datasets
    - If you manually input values, put them in their corresponding boxes
- Finally, push the "make predictions" button to display the predictions
