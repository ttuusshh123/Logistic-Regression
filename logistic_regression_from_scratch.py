*///import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('example.csv')

X = df[['sepal_length','petal_length']]

y = df['y']





class Logistic_Regression:
	def __init__(self, X, y):
		self.X = X#shape->mxn
		self.y = y#shape->mx1
		self.b = 0

	def sigmoid(self, z):
		return(1/(1+np.exp(-z)))

	def initial_W(self):
		self.W = np.zeros(((self.X).shape[1]))

	def z(self):
		return(np.dot(self.X,self.W) + self.b)

	def a(self):
		return(self.sigmoid(self.z()))
		
	def loss(self):
		a = self.a()
		return(-(self.y*np.log(a))-((1-y)*np.log(1-a)))#element wise product

	def predict(self, x):#shape of x is 1xn
		pred = (np.dot(x, self.W) + self.b)
		a = self.sigmoid(pred)
		return a
		# if(a >= 0.5): 
		# 	return 1
		# else:
		# 	return 0



		
		
		


	def fit_data(self, epochs = 200,lr = 0.001):
		
		self.initial_W()
		for i in range(epochs):
			z = self.z()
			a = self.a()
			self.W -= lr*(((np.dot((self.X).T,(a-y)))/y.size))
			self.b -= lr*((((a-y).sum())/y.size))
			print(f"loss after {i} epoch is {self.loss().mean()}")

			# print(f"weight updated {self.W}")
#*(self.X.shape[1])
lr  = Logistic_Regression(X, y)

lr.fit_data(10000, 0.3)

X_test = X[:50]
y_test = y[:50]
predictions = lr.predict(X_test)
# print(predictions)
predict_classes = []
for i in predictions:
	if i>=0.5:
		predict_classes.append(1)

	else:
		predict_classes.append(0)
		
print(accuracy_score(y_test, predict_classes))



# print(lr.W, lr.b)
# print(lr.W.shape)
# # print(lr.sigmoid(np.dot([[6.3, 5.1]],lr.W) + lr.b))
# print(lr.predict(x))
		
