# room-locator

Room-locator (WiFi receptor positioning system) is decision tree based machine learning program that determines the location of signal receptor out of 4 rooms. It takes in a dataset of a large number of samples each one containing 7 wifi signal strength values and the actual location of the receptor. It performs a cross-validation evaluation that splits the dataset into 10 folds and iteratively computes a decision tree for each fold and outputs averaged metrics (accuracy, precision, recall and F1 measure) of our program's predictions on the overall dataset. The program is also able to visualize decision trees and make predictions for new WiFi data.

# Installation

Before running the code make sure to have flask,numpy and matplotlib. In order for you to install them run the following commands in your terminal depending on the packages you need:
- ```pip install flask```
- ```pip install numpy```
- ```pip install matplotlib```

# General usage

To run the code and see results, run dashboard.py and follow the local host outputed in your terminal.
After copying the url in your web browser you can start running our program.

Our program has 3 main functionalities, all accessible from the webpage:
- Select the clean/noisy dataset or your own uploaded dataset for our model, and:
    - Evaluate the performance of our decision tree learning algorithm by performing a cross validation
    - Create a decision tree on the entire dataset
- Make predictions for new WiFi data (manually or uploaded) using the decision tree created
- Visualize the current and previously created decision trees

# Cross validation
- After choosing the training dataset, press the "Cross validation" button and it will show you both the confusion matrix and the relevant evaluation statistics:
       - the accuracy
       - the recall
       - the precision
       - the F1-measure

# Creating decision tree
- Before drawing decision tree or using the model for prediction, you first need to train the model on the provided dataset. For that, make sure to press the "Create decision tree" button and see a confirmation message
- If you want to clear the decision tree, press the "Reselect dataset" button

# Visualization
- Simply use the "draw decision tree" button on the right hand side and a number of graphs will be created representing your decision tree
    - The tree will be drawn across multiple plots and divided into subtrees in order for every node to be properly displayed.
    - If you're not able to properly see the information of each split point, make sure to zoom in or open the corresponding image in the "static/plots/" directory to see it closely
- To see the graphs created for previous decision trees, click the "View graohic history" link and a new page containing the visualization history will be launched. Our history keeps graphs for the most recent 9 decision trees, and older ones will be automatically deleted.

# Prediction
- If you want to test the decision tree with new data, you can either upload a file or manually input values for the different attributes:
    - If you upload a file make sure that it clears the following condition:
        - It's a .txt file
        - It's located inside the provided "predictions" directory
        - It has the same formatting as the provided clean and noisy datasets except for the labels
    - If you manually input values, put them in their corresponding boxes
- Finally, push the "make predictions" button to display the predictions
