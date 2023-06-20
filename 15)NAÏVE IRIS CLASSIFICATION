from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
gnb = GaussianNB()
mnb = MultinomialNB()

y_pred_gnb = gnb.fit(iris.data, iris.target).predict(iris.data)

cnf_matrix_gnb = confusion_matrix(iris.target, y_pred_gnb)

print(cnf_matrix_gnb)
y_pred_mnb = mnb.fit(iris.data, iris.target).predict(iris.data)
cnf_matrix_mnb = confusion_matrix(iris.target, y_pred_mnb)
print(cnf_matrix_mnb)
