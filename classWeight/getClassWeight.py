import pandas as pd


# 1. naive method
# cw = 1-(label/sum(label))
def get_naive_classWeight(df, label_column):
	label_sum = len(df)
	unique_label = df[label_column].unique()
	classWeight = {}
	for label in unique_label:
		weight = 1- (len(df[df[label_column] == label])/label_sum)
		classWeight.update({label:weight})
	return classWeight


# 2. balanced class weight
# cw = sum(label) / (n_class * label) 
# ref: https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/,
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
def get_balanced_classWeight(df, label_column):
	label_sum = len(df)
	unique_label = df[label_column].unique()
	n_class = len(unique_label)
	classWeight = {}
	for label in unique_label:
		weight = label_sum / (n_class * len(df[df[label_column] == label]))
		classWeight.update({label:weight})
	return classWeight