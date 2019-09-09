from ta import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



#read stock prices from file :
data = pd.read_csv("aapl.us.txt")
print(data.tail(2))


#define our X variables :
#add technical analysis features filling Nans values
data = add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)
print(data.tail(2))


#define our Y variable :
#evaluating the percentage of change in stock prices
data["diff"]=np.log(data["Close"].shift(1))-np.log(data["Close"])

target='diff'
#drop features corresponding to current stock price (e.g. Open, High, Low, Close, Volume)
features=['momentum_ao', 'momentum_mfi', 'volume_adi', 'volume_em', 'volatility_bbhi', 'volatility_bbli', 'trend_adx']
columns = np.union1d(['diff', 'Close'], features)
data = data[columns]



#plot heatmap of feature correlation :
corr = data.corr().abs()  #construction of correlation matrix
#generate a mask to ignore upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, mask=mask, cmap='coolwarm', fmt='.2f')
plt.show()


#drop rows and columns with NaN :
data.dropna(axis=1, how=any, thresh=100, inplace=True)  #first drop columns that mostly have NaN
data.dropna(axis=0, inplace=True)  #remove NaN from the shift(1) operation
print(data.tail(2))

#calculate ROI for buy low sell high trading strategy :
def calculate_roi(prediction):
    close=data['Close'].tail(len(prediction)).values
    investment=10000.00
    cash=investment
    stocks=0

    for i in range(len(prediction)):
        if prediction[i]>0 and cash>close[i]:
            stocks=stocks+cash/close[i]
            cash=cash%close[i]
        elif prediction[i]<0 and stocks>0:
            cash=cash+close[i]*stocks
            stocks=0

    returns=cash+close[-1]*stocks #cash+last stock price x stocks
    return 100*(returns/investment-1.0)



#splitting data for training and testing :
X=data[features]
y=data[target]
X_train,X_test,y_train,y_test=train_test_split(X.values, y.values,test_size=0.20,shuffle=False,random_state=123)

result=pd.DataFrame(columns=["actual","sklearn","statsmodel"])
result['actual']=y_test


#prediction using sklearn (LinearRegression) :
sklearn_lr_model=LinearRegression()
sklearn_lr_model.fit(X_train, y_train)
score=sklearn_lr_model.score(X_test, y_test)

result['sklearn']=sklearn_lr_model.predict(X_test)
print("LinearRegression",calculate_roi(result['sklearn']))



#prediction using Stats model :
ols=sm.regression.linear_model.OLS(y_train,X_train)
sm_ols_model=ols.fit()
predictions=sm_ols_model.predict(X_test)

result['statsmodel']=predictions
print("OrdinaryLeastSquares",calculate_roi(result['statsmodel']))

#plot actual vs prediction graph :
result.plot(color=['green','red','blue'])
plt.show()





