import pandas as pd
from pandas import Series, DataFrame

import numpy as np

# Plotting library
import matplotlib.pyplot as plt

# Data visualization library
import seaborn as sns
sns.set_style('whitegrid')

import re


# Machine Learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class Titanic:

	# Extracts a title if one is found
	def get_title(self, full_name):
		found_title = re.search(' ([A-Za-z]+)\.', full_name)
		if found_title:
			return found_title.group(1)
		return ""

	# Let's do some feature engineering + clean up the data
	def preprocess(self, train, test):
		data = [train, test]
		# First, let's look at if they have a cabin or not 
		train['HasCabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
		test['HasCabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

		# Modify input data to numerical values
		for d in data:
			# Add new feature, FamilySize, which is a combo of sibling/spouse and parent/children
			d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
			
			# Add new feature, IsAlone
			d['isAlone'] = np.where(d['FamilySize'] == 1, 1, 0)

			# Fill in the missing fare values with the mean
			d['Fare'] = d['Fare'].fillna(train['Fare'].median())

		# Split up fare prices into 3 seperate categories
		train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

		for d in data: 
			age_avg = d['Age'].median()
			age_std = d['Age'].std()
			age_nullcount = d['Age'].isnull().sum()

			# Generate random values for age depending on the avg and std
			age_random = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_nullcount)

			#d['Age'][np.isnan(d['Age'])] = age_random
			#d.loc[:, ('Age', np.isnan(d['Age']))] = age_random
			for age in age_random:
				d['Age'] = d['Age'].fillna(age)

		# Split up the age into 4 seperate categories
		train['CategoricalAge'] = pd.cut(train_df['Age'], 5)

		for d in data:
		 	d['Title'] = d['Name'].apply(self.get_title)

		#	# Group all boujee names int one special category 
		 	special_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
		 	d['Title'] = d['Title'].replace(special_titles, 'Special')

		 	# Replace French titles with English equivalent
		 	d['Title'] = d['Title'].replace('Mlle', 'Miss')
		 	d['Title'] = d['Title'].replace('Ms', 'Miss')
		 	d['Title'] = d['Title'].replace('Mme', 'Mrs')

		 	d['isMother'] = np.where((d['Sex'] == 'female') & (d['Age'] >18) & (d['Parch'] > 0) & (d['Title'] != 'Miss'), 1, 0)

		# Drop values we don't need anymore
		drop_vals = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
		train = train.drop(drop_vals, axis = 1)
		test = test.drop(drop_vals, axis = 1)
		train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
		data = [train, test]
		return data

	# Maps values to numerals as required by the algorithms
	def mapping(self, data):
		for d in data:
			# Map passenger class
			d['Pclass'].replace('3rd', 3, inplace = True)
			d['Pclass'].replace('2nd', 2, inplace = True)
			d['Pclass'].replace('1st', 1, inplace = True)
			
			# Map sex
			d['Sex'] = np.where(d['Sex'] == 'female', 0, 1)

			# Map titles
			title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
			d['Title'] = d['Title'].map(title_mapping)
			d['Title'] = d['Title'].fillna(0)
			d['Title'] = d['Title'].astype(int)

			# Map embarked
			d['Embarked'] = d['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
			d['Embarked'] = d['Embarked'].fillna(d['Embarked'].median())
			d['Embarked'] = d['Embarked'].astype(int)

			# Map the fare
			fare_med = d['Fare'].median()
			fare_low = fare_med / 2
			fare_high = fare_med * 2

			# Values were selected from CategoricalFare
			d.loc[d['Fare'] <= 7.91, 'Fare'] = 0
			d.loc[(d['Fare'] > 7.91) & (d['Fare'] <= 14.454), 'Fare'] = 1
			d.loc[(d['Fare'] > 14.454) & (d['Fare'] <= 31), 'Fare'] = 2
			d.loc[d['Fare'] > 31, 'Fare'] = 3
			d['Fare'] = d['Fare'].astype(int)

			# Map age: values were selected from CategoricalAge
			d.loc[d['Age'] <= 16, 'Age'] = 0
			d.loc[(d['Age'] > 16) & (d['Age'] <= 32), 'Age'] = 1
			d.loc[(d['Age'] > 32) & (d['Age'] <= 48), 'Age'] = 2
			d.loc[(d['Age'] > 48) & (d['Age'] <= 64), 'Age'] = 3
			d.loc[d['Age'] > 64, 'Age'] = 4
			d['Fare'] = d['Fare'].astype(int)

		return data

	# Visualize the features
	def visualize(self, train, feature):
		facet = sns.FacetGrid(train, hue='Survived', aspect=4)
		facet.map(sns.kdeplot, feature, shade=True)
		facet.set(xlim=(0, train[feature].max()))
		facet.add_legend()
		facet.savefig(feature + '1.png')

		fig, axis1 = plt.subplots(1, 1, figsize=(38, 10))
		avg = train[[feature, 'Survived']].groupby([feature], as_index=False).mean()
		sns.barplot(x=feature, y='Survived', data=avg)
		fig.savefig(feature + '2.png')

	def heatmap(self, train, name):
		colourmap = plt.cm.viridis
		plt.figure(figsize=(11,11))
		plt.title('Pearson Correlation of Features', y=1.55, size=15)
		sns_plot = sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colourmap, linecolor='white', annot=True)
		fig = sns_plot.get_figure()
		fig.savefig(name)

	def pairplot(self, train, name):
		graph = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked', u'FamilySize', u'Title']], 
			hue='Survived',
			palette='seismic',
			size=1.2,
			diag_kind='kde',
			diag_kws=dict(shade=True),
			plot_kws=dict(s=10) )

		graph.set(xticklabels=[])

		fig = graph.get_figure()
		fig.savefig(name)

	def logistic_regression(self, x_train, y_train, x_test):
		logreg = LogisticRegression()
		lr_fit = logreg.fit(x_train, y_train)
		y_pred = logreg.predict(x_test)
		score_lr = logreg.score(x_train, y_train)
		print score_lr
		return y_pred

	def svc(self, x_train, y_train, x_test):
		svc = SVC()
		svc.fit(x_train, y_train)
		y_pred = svc.predict(x_test)
		score_svc = svc.score(x_train, y_train)
		print score_svc
		return y_pred

	def random_forest(self, x_train, y_train, x_test):
		rf = RandomForestClassifier(n_estimators=100)
		rf.fit(x_train, y_train)
		y_pred = rf.predict(x_test)
		score_rf = rf.score(x_train, y_train)
		print score_rf
		return y_pred

	def k_neighbours(self, x_train, y_train, x_test):
		knn = KNeighborsClassifier(n_neighbors = 3)
		knn.fit(x_train, y_train)
		y_pred = knn.predict(x_test)
		score_knn = knn.score(x_train, y_train)
		print score_knn
		return y_pred

	def gaussian_nb(self, x_train, y_train, x_test):
		gnb = GaussianNB()
		gnb.fit(x_train, y_train)
		y_pred = gnb.predict(x_test)
		score_gnb = gnb.score(x_train, y_train)
		print score_gnb
		return y_pred

if __name__ == '__main__':
    titanic_ml = Titanic()

    # Import the test and training files
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    # Clean the data
    train_pcs, test_pcs = titanic_ml.preprocess(train_df, test_df)

    # Visualize some of the data
    #titanic_ml.visualize(train_pcs, 'Age')
    #titanic_ml.visualize(train_pcs, 'Fare')
    titanic_ml.visualize(train_pcs, 'FamilySize')

    # Map values to numerals
    train, test = titanic_ml.mapping([train_pcs, test_pcs])

    # Visualize the pre and post cleaning data
    titanic_ml.heatmap(train, 'postcleaning-hm.png')

    # Define training and testing set
    x_train = train.drop('Survived', axis=1)
    y_train = train['Survived']
    x_test = test.copy()

    # print x_train
    # print "************"
    # print y_train
    # print "************"
    # print x_test

    # Logistic Regression
    pred_lr = titanic_ml.logistic_regression(x_train, y_train, x_test)

    # SVC
    pred_svc = titanic_ml.svc(x_train, y_train, x_test)

    # Random Forest 
    pred_rf = titanic_ml.random_forest(x_train, y_train, x_test)

    # K-Neigbours
    pred_knn = titanic_ml.k_neighbours(x_train, y_train, x_test)

    # Gaussian Naive Bayes
    pred_gnb = titanic_ml.gaussian_nb(x_train, y_train, x_test)

    submission = pd.DataFrame({
    	'PassengerId': test_df['PassengerId'],
    	'Survived': pred_rf
    	})

    submission.to_csv('titanic_csv', index=False)



