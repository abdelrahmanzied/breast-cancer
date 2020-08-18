#Import
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


#Data
Data = load_breast_cancer()
X = Data.data
y = Data.target


#Polynomail Features
poly_reg = PolynomialFeatures()
newx = poly_reg.fit_transform(X)


#Spliiting
X_train, X_test, y_train, y_test = train_test_split(newx, y, test_size=.33, random_state=18, shuffle=True)


#Model
LogisticRegressionModel = LogisticRegression(penalty='l1', solver='liblinear',  C=1.0, random_state=18)
LogisticRegressionModel.fit(X_train, y_train)

print("Train Score: ", LogisticRegressionModel.score(X_train, y_train))
print("Test Score: ", LogisticRegressionModel.score(X_test, y_test)) 
print("Classes: ", LogisticRegressionModel.classes_) 
print("Number of itterations:  ", LogisticRegressionModel.n_iter_) 


#Predict
y_pred = LogisticRegressionModel.predict(X_test)
y_pred_prob = LogisticRegressionModel.predict_proba(X_test)
print('Real Values: ', y_train[:5])
print('Pred Values: ', y_pred[:5])


#Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n', CM)
sns.heatmap(CM, center=True)


#Accuracy Score
AccScore = accuracy_score(y_test, y_pred)
print('Accuracy Score: ', AccScore)