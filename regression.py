#_________________Import libraries and custom functions__________________

import numpy as np
from functions import *
import pandas as pd

#_______________________________Load data_________________________________

df=pd.read_csv('AirfoilSelfNoise.csv')
X = df[['f', 'alpha', 'c', 'U_infinity', 'delta']].values
y = df[['SSPL']].values

#________________Split data on training and test dataset___________________

X_train, y_train, X_test, y_test = train_test_split(X,y,0.3)

#___________________________Train the models________________________________

# Decision Tree Regressor
decision_tree=decision_tree_regression(X,y)


# Random Forest Regressor
random_trees=random_forest_regression(X,y)

# Gradient Boosting
gb_trees=gradient_boosting(X,y)

#____________________________Models evaluation_____________________________

# Decision Tree accuracy

dt_mse = DT_MSE(X_test, y_test,decision_tree)
print ("MSE for Decision Tree Algorithm is ", dt_mse)

# Random Forest accuracy

rf_mse = RF_MSE(X_test, y_test,random_trees)
print ("MSE for Random Forest Algorithm is ", rf_mse)

# Ada Boosting accuracy

gb_mse = GB_MSE(X_test,y_test,gb_trees)
print ("MSE for Gradient Boosting Algorithm is ", gb_mse)