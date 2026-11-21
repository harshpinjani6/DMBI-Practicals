import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset used in the Experiment 3 screenshots
# Place Movie_Streaming_Dataset.csv in the same folder.
df = pd.read_csv('Movie_Streaming_Dataset.csv')
print('Dataset Information:')
df.info()
print('\nFirst 5 Rows:')
print(df.head())
print('\nMissing Values:')
print(df.isnull().sum())

numeric = ['Age', 'Watch_Time', 'Rating', 'Total_Watch_Hours']
category = ['Gender', 'City', 'Genre', 'Language', 'Subscription_Type', 'Payment_Method']

print('\nMean:\n', df[numeric].mean())
print('\nMedian:\n', df[numeric].median())
print('\nMode:\n', df[category].mode().iloc[0])
print('\nStandard Deviation:\n', df[numeric].std())
print('\nVariance:\n', df[numeric].var())
print('\nRange:\n', df[numeric].max() - df[numeric].min())
q1 = df[numeric].quantile(0.25)
q3 = df[numeric].quantile(0.75)
print('\nInterquartile Range (IQR):\n', q3 - q1)
correlation = df[numeric].corr()
print('\nCorrelation Matrix:\n', correlation)

plt.figure(figsize=(10, 6))
sns.histplot(df['Watch_Time'].dropna(), kde=True)
plt.title('Distribution of Watch Time')
plt.xlabel('Watch Time (minutes)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Subscription_Type', y='Watch_Time')
plt.title('Boxplot of Watch Time by Subscription Type')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=1, fmt='.2f')
plt.title('Correlation Heatmap of Movie Streaming Attributes')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Watch_Time', y='Total_Watch_Hours', hue='Subscription_Type')
plt.title('Watch Time vs Total Watch Hours')
plt.tight_layout()
plt.show()
