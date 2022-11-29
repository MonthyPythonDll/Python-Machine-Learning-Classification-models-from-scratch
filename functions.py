
#________________________Import libraries_______________________________
import numpy as np
from random import randrange
from statistics import mean

#_______________________Functions______________________________________

# function that calculate mean squared error
def mean_squared_error(y):
	mse=0
	pred=np.mean(y)
	for i in range(len(y)):
		error=y[i]-pred
		mse+=error*error
	return mse/len(y)

# function that calculates mean value
def find_mean(y):
	return np.mean(y)

# function that calculates loss function for Decision Tree regressor
def loss_function_dt(left_labels, right_labels):
	n_labels=len(left_labels)+len(right_labels)

	prob_left=len(left_labels)/n_labels
	prob_right=len(right_labels)/n_labels

	loss=(prob_left*mean_squared_error(left_labels)+prob_right*mean_squared_error(right_labels))

	return loss

# function that find all potential splits of feature dataset
def splits(X):
	splits={}
	n_columns=X.shape[1]
	for column in range(n_columns):
		splits[column] = []
		values=X[:,column]
		unique_values=np.unique(values)

		for index in range (1,len(unique_values)):
			current_value = unique_values[index]
			previous_value = unique_values[index-1]
			split=(current_value+previous_value)/2
			splits[column].append(split)

	return splits

#Function that splits both labels and features
def data_split(X,y, column, value):
	data=np.column_stack((X,y))
	split_column=X[:,column]
	left_split_y=data[np.where([split_column<=value]),-1:]
	right_split_y=data[np.where([split_column>value]),-1:]
	left_split_x=data[np.where([split_column<=value]),:-1]
	right_split_x=data[np.where([split_column>value]),:-1]

	return left_split_x[1], right_split_x[1],left_split_y[1], right_split_y[1]

# function that find the best split of dataset based on minimal loss
def best_split(X,y):
	potential_splits=splits(X)
	loss=9999
	for column_index in potential_splits:
		for value in potential_splits[column_index]:
			_,_,left_data, right_data = data_split(X,y,column_index,value)
			current_loss=loss_function_dt(left_data,right_data)
			if current_loss<loss:
				loss=current_loss
				best_split_column=column_index
				best_split_value=value

	return best_split_column, best_split_value

#Decision Tree Regressor training function
def decision_tree_regression(X,y,depth=0, max_depth=3,min_samples=3):

	if (len(X)<min_samples) or (depth==max_depth):
		result=find_mean(y)
		return result		
	else:
		depth+=1
		possible_splits=splits(X)
		split_column,split_value=best_split(X,y)
		left_split_x, right_split_x,left_split_y, right_split_y= data_split(X,y, split_column, split_value)

		condition="{} <= {}".format(split_column, split_value)
		sub_tree={condition: []}

		true_node=decision_tree_regression(left_split_x,left_split_y, depth,max_depth, min_samples)
		false_node=decision_tree_regression(right_split_x,right_split_y, depth,max_depth, min_samples)

		if true_node==false_node:
			sub_tree=true_node
		else:
			sub_tree[condition].append(true_node)
			sub_tree[condition].append(false_node)

		return sub_tree


# Decision Tree Regressor predict function
def predict_result(input_data,model):
	condition=list(model.keys())[0]
	feature,_,value=condition.split()

	if input_data[int(feature)] <= float(value):
		result=model[condition][0]
	else:
		result=model[condition][1]

	if not isinstance(result,dict):
		return result
	else:
		residual_tree=result
		return predict_result(input_data,residual_tree)

# function that perform bootstraping
def sub_sampling(X,y,sample_size):
	dataset=np.column_stack((X,y))
	empty_array = np.array([], dtype=np.int64).reshape(0,X.shape[1]+1)
	sample_list=[]
	n_samples=int(len(dataset)*sample_size)
	for i in range(n_samples):
		index=randrange(len(dataset))
		if i==0:
			sample_array=np.vstack([empty_array, dataset[index].reshape(1,X.shape[1]+1)])
		else:
			sample_array=np.concatenate((sample_array, dataset[index].reshape(1,X.shape[1]+1)), axis=0)

	return sample_array[:,:-1], sample_array[:,-1:]

# Random Forest Regressor training function
def random_forest_regression(X,y,n_trees=20, sample_size=0.9, max_depth=3, min_samples=3):
	trees=[]
	for i in range(n_trees):
		X_sample, y_sample=sub_sampling(X,y,sample_size)
		tree=decision_tree_regression(X,y,max_depth=max_depth,min_samples=min_samples)
		trees.append(tree)
	return trees

# Random Forest Regressor predict function
def predict_result_random_forest(input_data,trees):
	results=[]
	for tree in trees:
		result=predict_result(input_data,tree)
		results.append(result)

	final_result=find_mean(results)

	return final_result

# Gradient Boosting training function
def gradient_boosting(X,y, max_depth=3, min_samples=3,n_steps=5):
	trees=[]
	for i in range(n_steps):
		tree=decision_tree_regression(X,y,max_depth=max_depth,min_samples=min_samples)
		for row in range(len(X)):
			prediction=predict_result(X[row],tree)
			residual=y[row]-prediction
			y[row]=residual
		trees.append(tree)

	return trees

# Gradient Boosting predict function
def grad_boost_predict(input_data,trees):
	predictions=[]
	for tree in trees:
		prediction=predict_result(input_data,tree)
		predictions.append(prediction)
	result=sum(predictions)

	return result

# function that calculates mean squared error of test set for Decision Tree Regressor
def DT_MSE(X_test,y_test,tree):
	total_error=0
	for i in range(len(X_test)):
		prediction=predict_result(X_test[i],tree)
		error=prediction-y_test[i]
		total_error+=error*error
	mse=total_error/len(y_test)
	return mse

# function that calculates mean squared error of test set for Random Forest Regressor
def RF_MSE(X_test,y_test,trees):
	total_error=0
	for i in range(len(X_test)):
		prediction=predict_result_random_forest(X_test[i],trees)
		error=prediction-y_test[i]
		total_error+=error*error
	mse=total_error/len(y_test)
	return mse

# function that calculates mean squared error of test set for Gradient Boosting
def GB_MSE(X_test,y_test,models):
	total_error=0
	for i in range(len(X_test)):
		prediction=grad_boost_predict(X_test[i],models)
		error=prediction-y_test[i]
		total_error+=error*error
	mse=total_error/len(y_test)
	return mse

#Function that splits test and training dataset
def train_test_split(X,y,ratio):
	random_order = np.random.permutation(len(X))
	X,y = X[random_order], y[random_order]
	test_size=int(ratio*len(X))
	train_size=int(len(X)-test_size)
	X_train=X[:train_size,:]
	y_train=y[:train_size]
	X_test=X[-test_size:,:]
	y_test=y[-test_size:]

	return X_train, y_train, X_test, y_test

	



