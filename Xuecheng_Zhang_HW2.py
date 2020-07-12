# %load ../standard_import.txt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot
from sklearn import svm
#part a
np.random.RandomState(1) # Set the random seed
def generata_data(n,p):
    n1=np.random.binomial(n,0.5)
    n2=100-n1
    x1=np.random.standard_t(1,n1)+p
    x2=np.random.standard_t(1,n2)
    y1=np.repeat(1,n1)
    y2=np.repeat(-1,n2)
    x=np.concatenate((x1,x2))# put x1 and x2 in same x
    y=np.concatenate((y1,y2))# put y1 and y2 in same y
    x=x.reshape(-1,1)#reshape x from 2D to 1D
    return x,y
err1=[]#define an empty list for LDA
err2=[]#define an empty list for logistic regression
lr = skl_lm.LogisticRegression(solver='newton-cg')
lda = LinearDiscriminantAnalysis(solver='svd')
for i in range(100):
    X_train,Y_train = generata_data(100,1)
    X_test,Y_test = generata_data(100,1)
    #LDA
    lda.fit(X_train,Y_train)#use training data to fit the lda model
    test_error1=sum(lda.predict(X_test)!=Y_test)#get the test error
    err1.append(test_error1)#put the test error in err1 list
    #logistic regression
    lr.fit(X_train,Y_train)#use the training dara to fit the logsitic regression model
    test_error2=sum(lr.predict(X_test)!=Y_test)#get the test error
    err2.append(test_error2)#put the test error in err2 list
df= {'LDA':err1, 'LR':err2}
df = pd.DataFrame(data=df)#save them in data frame
#find the mean and variance
mean = df.mean()
var = df.var()
print("The mean of test error is",mean)
print("The variance of test error is",mean)
box_plot = df.plot.box()
#part b,u=2
np.random.RandomState(1) # Set the random seed
def generata_data(n,p):
    n1=np.random.binomial(n,0.5)
    n2=100-n1
    x1=np.random.standard_t(1,n1)+p
    x2=np.random.standard_t(1,n2)
    y1=np.repeat(1,n1)
    y2=np.repeat(-1,n2)
    x=np.concatenate((x1,x2))# put x1 and x2 in same x
    y=np.concatenate((y1,y2))# put y1 and y2 in same y
    x=x.reshape(-1,1)#reshape x from 2D to 1D
    return x,y
err1=[]#define an empty list for LDA
err2=[]#define an empty list for logistic regression
lr = skl_lm.LogisticRegression(solver='newton-cg')
lda = LinearDiscriminantAnalysis(solver='svd')
for i in range(100):
    X_train,Y_train = generata_data(100,2)
    X_test,Y_test = generata_data(100,2)
    #LDA
    lda.fit(X_train,Y_train)#use training data to fit the lda model
    test_error1=sum(lda.predict(X_test)!=Y_test)#get the test error
    err1.append(test_error1)#put the test error in err1 list
    #logistic regression
    lr.fit(X_train,Y_train)#use the training dara to fit the logsitic regression model
    test_error2=sum(lr.predict(X_test)!=Y_test)#get the test error
    err2.append(test_error2)#put the test error in err2 list
df= {'LDA':err1, 'LR':err2}
df = pd.DataFrame(data=df)#save them in data frame
#find the mean and variance
mean = df.mean()
var = df.var()
print("The mean of test error is",mean)
print("The variance of test error is",mean)
box_plot = df.plot.box()
#part b u=3
np.random.RandomState(1) # Set the random seed
def generata_data(n,p):
    n1=np.random.binomial(n,0.5)
    n2=100-n1
    x1=np.random.standard_t(1,n1)+p
    x2=np.random.standard_t(1,n2)
    y1=np.repeat(1,n1)
    y2=np.repeat(-1,n2)
    x=np.concatenate((x1,x2))# put x1 and x2 in same x
    y=np.concatenate((y1,y2))# put y1 and y2 in same y
    x=x.reshape(-1,1)#reshape x from 2D to 1D
    return x,y
err1=[]#define an empty list for LDA
err2=[]#define an empty list for logistic regression
lr = skl_lm.LogisticRegression(solver='newton-cg')
lda = LinearDiscriminantAnalysis(solver='svd')
for i in range(100):
    X_train,Y_train = generata_data(100,3)
    X_test,Y_test = generata_data(100,3)
    #LDA
    lda.fit(X_train,Y_train)#use training data to fit the lda model
    test_error1=sum(lda.predict(X_test)!=Y_test)#get the test error
    err1.append(test_error1)#put the test error in err1 list
    #logistic regression
    lr.fit(X_train,Y_train)#use the training dara to fit the logsitic regression model
    test_error2=sum(lr.predict(X_test)!=Y_test)#get the test error
    err2.append(test_error2)#put the test error in err2 list
df= {'LDA':err1, 'LR':err2}
df = pd.DataFrame(data=df)#save them in data frame
#find the mean and variance
mean = df.mean()
var = df.var()
print("The mean of test error is",mean)
print("The variance of test error is",mean)
box_plot = df.plot.box()

#question 3



#part a
vector_x1 = np.array([3, 2, 4, 1, 2, 4, 4])
vector_x2 = np.array([4, 2, 4, 4, 1, 3, 1])
matplotlib.pyplot.scatter(vector_x1,vector_x2,color=["r", "r", "r", "r", "b", "b", "b"])

#part b

X = np.vstack((vector_x1,vector_x2)).T#add vector_x1 and vector_x2 as column vector
Y = np.array([1]*4 + [0]*3)#"1" is red, and repect 4 times,"0"is blue which repect 3 times
clf = svm.SVC(kernel='linear',C=9999)#since optimal separating hyperplane, so we set Regularization parameter become a big number
clf.fit(X, Y)#fir X and Y in clf model
w = clf.coef_[0]# get the coefficients
a = -w[0] / w[1]# find the slope by negative beta1 divided beta 2
xx = np.linspace(-1, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]#create a relationship
matplotlib.pyplot.scatter(vector_x1,vector_x2,color=["r", "r", "r", "r", "b", "b", "b"])
plt.plot(xx, yy, 'k-')

#part c:  The classification rule is “Classify to Red if X1−X2−0.5<0, and classify to Blue otherwise.”
#part d: code based on Google
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# plot the line, the points, and the nearest vectors to the plane
X = np.vstack((vector_x1,vector_x2)).T#add vector_x1 and vector_x2 as column vector
Y = np.array([1]*4 + [0]*3)#"1" is red, and repect 4 times,"0"is blue which repect 3 times
clf = svm.SVC(kernel='linear',C=9999)#since optimal separating hyperplane, so we set Regularization parameter become a big number
clf.fit(X, Y)#fir X and Y in clf model
w = clf.coef_[0]# get the coefficients
a = -w[0] / w[1]# find the slope by negative beta1 divided beta 2
xx = np.linspace(-1, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]#create a relationship
matplotlib.pyplot.scatter(vector_x1,vector_x2,color=["r", "r", "r", "r", "b", "b", "b"])
# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()#the decision function of x
ylim = ax.get_ylim()#the decision function of y

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T#add XX.ravel() and YY.ravel()as column vector
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])# plot decision boundary and margins
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()# plot support vectors



#part e:The support vectors are the points (2,1), (2,2), (4,3) and (4,4).
#part f:if we moved the observation (4,1),(1,4),(3,4)[they are not the support vectors] we would not change the maximal margin hyperplane.


#part g
#code based on Google
#https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
X = np.vstack((vector_x1,vector_x2)).T#add vector_x1 and vector_x2 as column vector
Y = np.array([1]*4 + [0]*3)#"1" is red, and repect 4 times,"0"is blue which repect 3 times
clf = svm.SVC(kernel='linear')#since optimal separating hyperplane, so we set Regularization parameter become a big number
clf.fit(X, Y)#fir X and Y in clf model
w = clf.coef_[0]# get the coefficients
a = -w[0] / w[1]# find the slope by negative beta1 divided beta 2
xx = np.linspace(-1, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]#create a relationship
matplotlib.pyplot.scatter(vector_x1,vector_x2,color=["r", "r", "r", "r", "b", "b", "b"])
# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()
#part h
matplotlib.pyplot.scatter(vector_x1,vector_x2,color=["r", "r", "r", "r", "b", "b", "b"])
plt.plot(xx, yy, 'k-')
plt.scatter(1.2,3.9,color="b")#this number should be red, but I set it to be blue
