# Naive-Bayes Classifier for Email Spam Filtering
## Author: Matthew Jeremy

The purpose of this project is to build a machine learning model using Naive-Bayes classification that is capable of 
determining and predicting whether a particular email is classified as spam or otherwise (i.e. ham). 
The input data to this model are emails in the form of text files that can be found in the "./datasets" folder which
consists of two sub-folders corresponding to a "train" dataset for training the model as well as a "test" dataset for testing
the model against unseen data. This binary (two-class) classifier begins by processing the training dataset found in 
"./datasets/train" to calculate the spam and ham class probabilities based on the number of spam and ham emails that
exist in the training set as well as the conditional probabilities mapped to each word (also referred to as tokens) 
encountered in the emails. By using these calculated probabilities from the training dataset as a benchmark, the model is 
then able to read in a new unprocessed email and make an informed decision of predicting if it is spam or not spam. This 
model can also list down the most indicative spam and ham words based on the dataset it was trained with to see which 
tokens have the highest spam and ham indication values. This repository consists of: 
- **nb_classifier.py**: Main program that defines the class and functions that parse the input text file, creates and trains the model, tests for spam or otherwise, list down indicative words, and so forth. 
- **test1_script.py**: Executable script that passes the inputs to the main program and tests the trained model
- **datasets**: Folder that contains the training set and test set of emails/text files to pass as input to the classifier


### How to Run the "nb_classifier.py" Program: 
1. Download this repository onto your local machine

2. Start a new terminal session and access the directory that the repository is saved in **(MAKE SURE ALL FILES AND FOLDERS LISTED ABOVE ARE IN THIS SAME DIRECTORY)**

3. Start the program by entering the command: ```python nb_classifier.py``` then ```python``` to access the Python console

4. Execute the import statement: ```import nb_classifier as nb```

5. To create and train the classifier, enter the command: 

```model = nb.SpamFilter("datasets/train/spam", "datasets/train/ham", 1e-5)```

6. After processing the training dataset, the model can be tested against unseen emails in the "test" subfolder

7. Pass the path to the desired input text file to the function is_spam() as shown below, which will return "True" if it is classified as spam and return "False" otherwise
```# Passing test input file "dev1"```
```model.is_spam("datasets/test/ham/dev1")```
```# Passing test input file "dev201"```
```model.is_spam("datasets/test/spam/dev201")```

8. To determine the n most indicative words for each class in the "model" classifier, run the following commands: 
```# Lists the 5 words that have the highest spam indicator```
```model.most_indicative_spam(5)``` 
```# Lists the 3 words that have the highest ham indicator```
```model.most_indicative_ham(3)``` 


### How to Run the "test1_script.py" Test Script:
1. Download this repository onto your local machine

2. Start a new terminal session and access the directory that the repository is saved in

3. Run the command and see output: ```python test1_script.py```


