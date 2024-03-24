# Required Libraries
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error

#reading the dataset
data =pd.read_csv('Student_Performance.csv')

# Set display options to show all columns
pd.set_option('display.max_columns', None)

print(data.head(5))
print(data.shape)
print(data.columns)

#Dropping missing values 
data.dropna(how = 'any', inplace=True) 


#Remove Duplicates
data.drop_duplicates(inplace=True)

#Feature Encoding
LabelEncoder=LabelEncoder()
data['Extra_Activities_Encode'] = LabelEncoder.fit_transform(data['Extracurricular Activities'])
print(data)

#Finding Correlation
corr = data[['Hours Studied', 'Previous Scores', 'Extra_Activities_Encode',
       'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index']].corr()
print(corr)

#heatmap of corr
sns.heatmap(corr, annot=True)

#splitting the Data into independent and dependent variables
X = data[['Hours Studied', 'Previous Scores', 'Extra_Activities_Encode',
       'Sleep Hours', 'Sample Question Papers Practiced']]
y = data[['Performance Index']]

#splitting the Data into train and test datas
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Doing Regression
Linear_regression_model = LinearRegression()
Linear_regression_model.fit(x_train, y_train)

#Make prediction on the test data
y_pred_lr = Linear_regression_model.predict(x_test)

#Compute the R-Squared score
r2_score(y_test,y_pred_lr)
print(f'Performance of the model =  {round(r2_score(y_test,y_pred_lr),4)*100}%')

# Mean Absolute Error (MAE)
print(f'Mean Absolute Error =  {round(mean_absolute_error(y_test,y_pred_lr),4)*100}%')

#Root Mean Square Error (RMSE) and Mean Squared Error (MSE)
mse = round(mean_squared_error(y_test,y_pred_lr),2)
print(mse)
print(f'Root Mean Square Error = ' , mse ** (0.5))


################### THIS IS THE END ##### THANK YOU :) 