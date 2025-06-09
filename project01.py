import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df=pd.read_csv("global fashion1.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Ensure 'month' is ordered
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
               'August', 'September', 'October', 'November', 'December']
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

monthly_sales = df.groupby('month')['sales'].sum().reindex(month_order)


plt.figure(figsize=(12,6))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o')
plt.title('Total Sales by Month (Seasonality)')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='month', y='sales', data=df, order=month_order)
plt.title('Sales Distribution by Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,7))
sns.lineplot(data=df, x='month', y='sales', hue='region', estimator='sum', ci=None, marker='o')
plt.title('Monthly Sales by Region')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# Linear regression model

monthly_sales = df.groupby('month')['sales'].sum().reindex(month_order).reset_index()
monthly_sales['month_num'] = monthly_sales.index + 1  # January=1, ..., December=12

# Linear regression
X = monthly_sales[['month_num']]
y = monthly_sales['sales']
model = LinearRegression()
model.fit(X, y)

# Predict sales for each month
predicted = model.predict(X)

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(monthly_sales['month'], y, marker='o', label='Actual')
plt.plot(monthly_sales['month'], predicted, marker='x', linestyle='--', label='Predicted')
plt.title('Linear Regression: Actual vs Predicted Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.legend()
plt.tight_layout()







