import numpy as np 
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime

## Forecast done 30 days by 30 days, as the model is incapable of doing
## 356 days at one time. This is more accurate. 

## Retrieve Data
df2 = pd.read_csv('SampleSubmission.csv')
df2 = df2[["Date"]]
final_len = len(df2)
df = pd.read_csv("StockPriceData.csv")

## Remove all other data and leave the close column
df = df[["Close"]]

## Create a list for the final values to be collected in
final = []

## Forecast variable made that can be altered depending on the number
## of days we wish to forecast
forecast = 30

## Create a prediction column that introduces a forecast number of days lag
df['Prediction'] = df[['Close']].shift(-forecast)
## print(df)

for i in range(12):
## print(df)

## Create an independent data set (X) 
    X = np.array(df.drop(['Prediction'],1))

## Remove the last '30' rows
    X = X[:-forecast]

## Create an dependent data set (y)  #####
    y = np.array(df['Prediction'])

## Get all of the y values except for the last '30' rows
## Split data into 75% training and 25% testing
    y = y[:-forecast]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

## print("start of trains")
## print(x_train)
## print(y_train)
## print("end of trains")
    
## Create and train the Support-Vector-Machine 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)


## Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
## The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)
    #print("svm confidence: ", svm_confidence)
    
## Remove the fields with NA in them, can be done with dropna too
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast:]

## Predict the next forecast number of days
    svm_prediction = svr_rbf.predict(x_forecast)

## Converting the array to a list for easier processing
    svm_prediction = svm_prediction.tolist()

## Add the prediction to the final list
    final.extend(svm_prediction)

## Prepare data for next round of proessing
    temp2 = y.tolist()
    temp_close = temp2 + svm_prediction

## Convert back to a dataframe
    df = pd.DataFrame()
    df["Close"] = temp_close
    df['Prediction'] = df[['Close']].shift(-forecast)

#print(final)

final = final[:final_len]

df2["Close"] = pd.DataFrame(final)
print(df2)

df2.to_csv(r'SVM.csv', index=False, header=True)

    
    

