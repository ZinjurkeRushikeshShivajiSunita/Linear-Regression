# Linear-Regression
iNeuron Assignment

# Build the linear regression model using scikit learn in boston data to predict 'Price' based on other dependent variable.

 Importing Library
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline

 Boston House price prediction dataset is loaded from sklearn library with the help of pandas 

from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)
bos

 Feature Names
boston.feature_names

 Data is Stored in data Frame
data = pd.DataFrame(data=data, columns=boston.feature_names )
data.head()

To check Correlation between dataset plot a pairplot /DataVisualization

sns.pairplot(data)

rows=2
cols=7
fig, ax=plt.subplots(nrows=rows,ncols=cols,figsize=(16,4))
col=data.columns
index=0

for i in range (rows):
    for j in range(cols):
        sns.distplot(data[col[index]],ax=ax[i][j])
        index=index+1
        
plt.tight_layout()

Feature Selection is done based on correlation between dataset
def getCorrelatedFeatures(corrdata,threshold):
    feature=[]
    value=[]
    
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index])>threshold:
            feature.append(index)
            value.append(corrdata[index])
        
    df = pd.DataFrame(data=value , index=feature, columns=['corr values'])
    return df
stored the feature in getCorrelatedFeatures
only ['RM','PTRATIO','LSTAT',''Price']feature shows correlation 


Model Training

model=LinearRegression()
model.fit(X_train,y_train)
y_predict=model.predict(x_test)
y_predict

df=pd.DataFrame(data=[y_predict,y_test])
df.T

 r2_score
score=r2_score(y_test,y_predict)
MAE=mean_absolute_error(y_test,y_predict)
MSE=mean_squared_error(y_test,y_predict)

print('r2_score:', score)
print('MAE:',MAE)
print('MAE:',MSE)

//Accuracy 

r2_score: 0.48816420156925056
MAE: 4.404434993909258
MAE: 41.67799012221684




    


