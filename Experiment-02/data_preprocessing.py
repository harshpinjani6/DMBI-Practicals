import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset from the same folder.
df = pd.read_csv('Movie_Streaming_Dataset.csv')
print('Original Dataset:')
print(df)

print('\nDataset Shape:')
print(df.shape)
print('\nDataset Information:')
df.info()

print('\nMissing Values:')
print(df.isnull().sum())

# Handle missing values (mean for numerical, mode for categorical).
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['City'] = df['City'].fillna(df['City'].mode()[0])
df['Payment_Method'] = df['Payment_Method'].fillna(df['Payment_Method'].mode()[0])
print('\nMissing Values After Handling:')
print(df.isnull().sum())

print('\nDuplicate Records:')
print(df[df.duplicated()])
df = df.drop_duplicates().copy()
print('\nDataset After Removing Duplicates:')
print(df)

categorical_columns = [
    'Viewer_ID', 'Viewer_Name', 'Gender', 'City', 'Movie_ID',
    'Movie_Title', 'Genre', 'Language', 'Subscription_Type',
    'Payment_Method'
]
for column in categorical_columns:
    df[column] = LabelEncoder().fit_transform(df[column])
print('\nDataset After Encoding:')
print(df)

numerical_columns = ['Age', 'Watch_Time', 'Rating', 'Total_Watch_Hours']
df[numerical_columns] = StandardScaler().fit_transform(df[numerical_columns])
print('\nDataset After Feature Scaling:')
print(df)

print('\nFinal Dataset:')
print(df)
print('\nMissing Values:')
print(df.isnull().sum())
print('\nNumber of Duplicate Records:')
print(df.duplicated().sum())
