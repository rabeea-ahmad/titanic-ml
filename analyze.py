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
		print "Reading test data"

		# Read in the training set csv
		data = pd.read_csv(infile)
		print "Finished reading data"

		# Clean up data and fill in any missing values
		print "Cleaning up data"
		median_age = data['Age'].median()
		data['Age'].fillna(median_age, inplace = True)

		# Extract the data we want = pclass, age, sex
		data_input = data[["Pclass", "Age", "Sex"]]
		#print data_input

		expected_output = data[["Survived"]]
		#print expected_output.head()

		# Modify input data to proper numbers to feed into algorithm
		# Pclass: 3rd = 3, 2nd = 2, 1st = 1
		# Sex: female = 0, male = 1
		data_input["Pclass"].replace("3rd", 3, inplace = True)
		data_input["Pclass"].replace("2nd", 2, inplace = True)
		data_input["Pclass"].replace("1st", 1, inplace = True)
		data_input["Sex"] = np.where(data_input["Sex"] == "female", 0, 1)

		return [data_input, expected_output]

	def split(self, input, expected):
		print "Split the training data"
		x_train, x_test, y_train, y_test = train_test_split(input, expected, train_size = 0.33, random_state = 42)
		return [x_train, x_test, y_train, y_test]

	def train(self, x_train, x_test, y_train, y_test):
		print "Training the data using the Random Forest Classifier algorithm"
		rf = RandomForestClassifier(n_estimators = 100)

		# Fit function trains the data
		rf.fit(x_train, y_train.values.ravel())

		# Check the accuracy
		acc = rf.score(x_test, y_test)
		print "The model accuracy: %", acc * 100

		# Write the model to a file
		joblib.dump(rf, "titanic_model1", compress=9)



		

if __name__ == '__main__':
    titanic_ml = Titanic()
    input, inputy = titanic_ml.preprocess('./data/train.csv')
    x_train, x_test, y_train, y_test = titanic_ml.split(input, inputy)
    titanic_ml.train(x_train, x_test, y_train, y_test)




