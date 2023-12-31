# room-locator

Room-locator (WiFi receptor positioning system) is decision tree based machine learning program that determines the location of signal receptor out of 4 rooms. It takes in a dataset of a large number of samples each one containing 7 wifi signal strength values and the actual location of the receptor. It performs a cross-validation evaluation that splits the dataset into 10 folds and iteratively computes a decision tree for each fold and outputs averaged metrics (accuracy, precision, recall and F1 measure) of our program's predictions on the overall dataset. The program is also able to visualize datasets, draw decision trees, and make predictions for new WiFi data. Finally, customizable pruning optimization is also implemented in our program.

# Installation

Before running the code make sure to have flask,numpy and matplotlib. In order for you to install them run the following commands in your terminal depending on the packages you need:
- ```pip install flask```
- ```pip install numpy```
- ```pip install matplotlib```

# General usage

To run the code and see results, run dashboard.py and follow the local host outputed in your terminal.
After copying the url in your web browser you can start running our program.

Our program has 4 main functionalities, all accessible from the webpage:
- Select the clean/noisy dataset or your own uploaded dataset for our model, and:
    - Visualize the WiFi dataset distribution by class in 2D representation
    - Evaluate the performance of our decision tree learning algorithm by performing a cross validation
    - Create a decision tree on the entire dataset
- Make predictions for new WiFi data (manually or uploaded) using the decision tree created
- Visualize the current and previously created decision trees
- Pruning decision tree optimization

# Visualize WiFi dataset
- Clean and Noisy dataset already exist in "wifi_db/" directory, but you can also put your own dataset in the same directory for selection. After selecting which dataset to be used for training our model, click on "Draw dataset in 2D" below to see a color coded scatter plot of the WiFi dataset. High dimensional data is transformed to 2D polar coordinate, which enables visualization, and the distribution patterns of data points could be roughly observed.
- To see the history of previous (non-repeated) plots, click on "View graphic history". We have already created visualizations for Clean and Noisy Datasets. Graphs are stored in "static/wifi_visual/" directory.

# Cross validation
- When the training dataset is selected, press the "Cross validation" button and it will show you both the confusion matrix, overall prediction accuracy, and the relevant evaluation statistics by class:
       - the recall
       - the precision
       - the F1-measure

# Creating decision tree
- Before drawing decision tree or using the model for prediction, you first need to train the model on the provided dataset. For that, make sure to press the "Create decision tree" button and see a confirmation message
- If you want to clear the decision tree, press the "Reselect dataset" button

# Visualize decision tree
- Simply use the "draw decision tree" button on the right hand side and a number of graphs will be created representing your decision tree
    - The tree will be drawn across multiple plots and divided into subtrees in order for every node to be properly displayed.
    - If you're not able to properly see the information of each split point, make sure to zoom in or open the corresponding image in the "static/plots/" directory to see it closely
- To see the graphs created for previous decision trees, click the "View graohic history" link and a new page containing the visualization history will be launched. Our history keeps graphs for the most recent 9 decision trees, and older ones will be automatically deleted. Graphs are stored in "static/plots/" directory.
- Note: we have plotted decision trees for Clean and Noisy (pruned) Datasets, both are visible in graphic history.

# Prediction
- There's a sample dataset in the "predictions/" directory, and you can put your own data in the same directory for selection. To make predictions on which room a WiFi receptor is in using our decision tree, you can either upload a file or manually input values for the different attributes:
    - If you upload a file make sure that it clears the following condition:
        - It's a .txt file
        - It's located inside the provided "predictions" directory
        - It has the same formatting as the provided clean and noisy datasets except for the labels
    - If you manually input values, put them in their corresponding boxes
- Finally, push the "make predictions" button to display the room numbers

# Pruning optimization
- Our program enables customized pruning parameter selection, and users can input 2 vaules: Depth limit (the starting depth from which pruning is considered, allowed values 2 - 15), and Dominant label percentage limit (decision tree algorithm will stop when the subset of data has a dominating label reaching certain percentage, allowed value 50-100)
- If only one value is entered, the other will be set to default (Depth 7 or Percentage 95)
- To apply pruning on decision tree learning, simple enter the values in the boxes before cross validation or creating decision trees
