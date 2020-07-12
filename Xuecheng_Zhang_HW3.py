import  numpy  as np
import  matplotlib.pyplot  as plt
from pygam import LogisticGAM ## LogisticGAM is for classification.



def  generate_data(n, p):
    #n1 is the data size for Y=1
    n_1 = np.random.binomial(n,0.5)
    #n0 is the daa size for Y=0
    n_0 = n-n_1
    #set the x1 and x0 
    x_1=np.zeros((n_1,p))
    x_0=np.zeros((n_0,p))
    #case 1
    for i in range (p):
        if (i+1)%2==0:
            x_1[:,i]=np.random.standard_t(1,n_1)+2
        else:
            x_1[:,i]=np.random.exponential(1,n_1)
    #case 2
    for i in range (p):
        if (i+1)%2==0:
            x_0[:,i]=np.random.standard_t(1,n_0)
        else:
            x_0[:,i]=np.random.exponential(1/3,n_0)
    #merge x1,x0
    X=np.vstack((x_1,x_0))
    Y=np.repeat([1,0],(n_1,n_0))
    return X,Y
    #define an empty list for error term
def simulation(No_T,n,p,box_plot=True):
    err=[]
    for i in range (No_T):
    #generate the test data
        X_train,Y_train=generate_data(n,p)
        X_test,Y_test= generate_data(n,p)
        
        logit_gam = LogisticGAM()
        logit_gam.gridsearch(X_train,Y_train)
        
        #calculate test error
        test_err=sum(logit_gam.predict(X_test)!=Y_test)/n
        err.append(test_err)
    if box_plot:
        plt.figure(num=None,figsize=(8,6),dpi=80)
        plt.boxplot(err)
        plt.text(1.1,0.15,"Mean:{:.2f}".format(np.mean(err)))
        plt.text(1.1,0.14,"Var:{:.3f}".format(np.var(err)))
        plt.title("logisticGAM")
        plt.ylabel("Test Error")
        plt.show()
simulation(100,100,10)
simulation(100,100,30)
        

        
        
        
        