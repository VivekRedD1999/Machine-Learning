import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df=pd.read_csv("E:\\files\\pdf\\ML\\datasheets\\Salary_Data.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
m,c=np.polyfit(df["Salary"],df["YearsExperience"],1)
plt.scatter(df["YearsExperience"],df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.plot(df["YearsExperience"],df["Salary"],color="red",linewidth=2)
plt.show()
plt.plot(df["Salary"],df["YearsExperience"])
plt.plot(df["Salary"],m*df["Salary"]+c,'r')
plt.show()
x=df[["YearsExperience"]]
y=df[["Salary"]]
x_train, x_test, y_train, y_test = train_test_split(x,y)
lr=LinearRegression()
lr.fit(x_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
y_pred=lr.predict(x_test)
print("Accuracy is")
print(r2_score(y_test,y_pred)*100)
print(x_test,y_pred)
