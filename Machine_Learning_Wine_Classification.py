import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

print ('-------------------- WINE CLASSIFICATION ----------------------')

# Load Dataset
wine_dataset = pd.read_csv('C:/Users/Luiz Fernando/Desktop/Projects/wine_dataset.csv')

# Change text to number for the network gets to do calculate
# 0 = Red wine
# 1 = white wine
wine_dataset['style'] = wine_dataset['style'].replace('red', 0)
wine_dataset['style'] = wine_dataset['style'].replace('white', 1)
#print(wine_dataset.head(20))

# Separate the variables between predictors and target variable
y = wine_dataset['style']
x = wine_dataset.drop('style', axis=1) 

# Creating the train and test variables
# test_size: use to pass database percentage that will be use for test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Model creating
# ExtraTreesClassifier is a machine learning aalgorithm-> build decision trees
model = ExtraTreesClassifier()
# Apply the algorithm to data
model.fit(x_train, y_train) 

# Print results
# Algorithm receive test data (t_test) and do a decision
# After that, it is done a prevision and compare with real results (y_test). Discover which % of correct answers
results = model.score(x_test, y_test)
print('Accuracy: ', results) 

# Making a random prediction
testset_data = x_test[400:403]
testset_styles = np.array(y_test[400:403])

predict = model.predict(testset_data)
print('Predictions are:')
print(predict)

print('Correct styles are:')
print(testset_styles)

#Calculation to discover % of hits
percent_hits = 0
for i in predict:
    if (predict[i] == testset_styles[i]):
        percent_hits = percent_hits + (1/len(predict))

percent_hits = percent_hits*100
print('Algorithm had', percent_hits, '% of hits in this test!')
