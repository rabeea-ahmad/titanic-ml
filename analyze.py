import numpy as np
import pandas as pd

# Randomf Forest Classifier  =  machine learning algorithm
from sklearn.ensemble import RandomForestClassifier

# Test train split
from sklearn.cross_validation import train_test_split

# Switch off pandas warning triggers
pd.options.mode.chained_assignment = None

# Used to write our model to a file
from sklearn.externals import joblib

class Titanic:


	@staticmethod
	def print_full(x):
	    pd.set_option('display.max_rows', len(x))
	    print(x)
	    pd.reset_option('display.max_rows')


	def preprocess(self, infile):
		print "preprocessing"

		# Step 1: Read in the training set csv
		data = pd.read_csv(infile)
		print "Finished reading data"
		print(data.head())

		#print(data.columns)

		# Step 2: Choose which columns are important to look at: passenger class, age, sex
		# Step 3: Clean up data and fill in any missing values

		#print(data['age'].head())


if __name__ == '__main__':
    titanic_ml = Titanic()
    titanic_ml.preprocess('./data/train.csv')




