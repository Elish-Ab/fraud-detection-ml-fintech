import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

import hashlib
# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to handle missing values
def handle_missing_values(df):
    return df.dropna()

# Function to remove duplicates
def remove_duplicates(df):
    return df.drop_duplicates()

# Function to correct data types
def correct_data_types(df):
    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')

    # Extract useful features from the 'signup_time' and 'purchase_time' columns
    df['signup_year'] = df['signup_time'].dt.year
    df['signup_month'] = df['signup_time'].dt.month
    df['signup_day'] = df['signup_time'].dt.day
    df['signup_hour'] = df['signup_time'].dt.hour
    df['purchase_year'] = df['purchase_time'].dt.year
    df['purchase_month'] = df['purchase_time'].dt.month
    df['purchase_day'] = df['purchase_time'].dt.day
    df['purchase_hour'] = df['purchase_time'].dt.hour

    return df


# Function to perform exploratory data analysis
def exploratory_data_analysis(df):
    print("Data Summary:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Transaction count
    print("\nTransaction Count:", len(df))
    
    # Average fraud and non-fraud transaction values
    avg_fraud = df[df['class'] == 1]['purchase_value'].mean()
    avg_non_fraud = df[df['class'] == 0]['purchase_value'].mean()
    print("\nAverage Fraud Transaction Value:", avg_fraud)
    print("Average Non-Fraud Transaction Value:", avg_non_fraud)
    
    # Fraud rate by hour
    df['transaction_hour'] = df['purchase_time'].dt.hour
    fraud_hourly = df.groupby('transaction_hour')['class'].mean()
    
    # Visualizations
    plt.figure(figsize=(10, 5))
    sns.countplot(x='class', data=df)
    plt.title('Fraud (1) vs Non-Fraud (0) Transactions')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='class', y='purchase_value', data=df)
    plt.title('Transaction Value Distribution by Fraud Status')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.histplot(df['purchase_value'], bins=50, kde=True)
    plt.title('Distribution of Purchase Value')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=fraud_hourly.index, y=fraud_hourly.values)
    plt.title('Fraud Rate by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Fraud Rate')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    fraud_daywise = df.groupby(df['purchase_time'].dt.dayofweek)['class'].mean()
    sns.barplot(x=fraud_daywise.index, y=fraud_daywise.values)
    plt.title('Fraud Rate by Day of the Week')
    plt.xlabel('Day of the Week (0=Monday)')
    plt.ylabel('Fraud Rate')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.show()

# Function to merge datasets for geolocation analysis
def find_country(ip, ip_df):
    # Ensure ip_df is sorted by lower_bound_ip_address
    ip_df = ip_df.sort_values(by='lower_bound_ip_address')
    row = ip_df[(ip_df['lower_bound_ip_address'] <= ip) & (ip_df['upper_bound_ip_address'] >= ip)]
    return row['country'].values[0] if not row.empty else 'Unknown'

def merge_datasets(fraud_df, ip_df):
    # Ensure data types are correct
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)

    # Apply range-based lookup for each row
    fraud_df['country'] = fraud_df['ip_address'].apply(lambda ip: find_country(ip, ip_df))
    
    # Optionally handle missing values in 'country'
    fraud_df['country'].fillna('Unknown', inplace=True)
    
    return fraud_df


# Function to engineer features
def feature_engineering(df):
    df['transaction_hour'] = df['purchase_time'].dt.hour
    df['transaction_day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_diff_signup_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    df['is_night_transaction'] = df['transaction_hour'].apply(lambda x: 1 if 0 <= x < 6 else 0)
    df['fraud_rate_by_browser'] = df.groupby('browser')['class'].transform('mean')
    df['fraud_rate_by_source'] = df.groupby('source')['class'].transform('mean')
    return df

# Function to visualize new features
def visualize_new_features(df):
    # Visualize time difference between signup and purchase
    plt.figure(figsize=(10, 5))
    sns.histplot(df['time_diff_signup_purchase'], bins=50, kde=True)
    plt.title('Distribution of Time Difference (Signup to Purchase in Hours)')
    plt.xlabel('Hours')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='class', y='time_diff_signup_purchase', data=df)
    plt.title('Time Difference (Signup to Purchase) by Fraud Status')
    plt.xlabel('Fraud (1) vs Non-Fraud (0)')
    plt.ylabel('Hours')
    plt.show()
    
    # Visualize night transactions count
    plt.figure(figsize=(10, 5))
    sns.countplot(x='is_night_transaction', data=df)
    plt.title('Count of Night vs. Day Transactions')
    plt.xlabel('Is Night Transaction (1=Yes, 0=No)')
    plt.show()
    
    # Fraud rate by browser visualization - bar plot
    plt.figure(figsize=(12, 6))
    fraud_rate_browser = df.groupby('browser')['class'].mean().sort_values(ascending=False)
    sns.barplot(x=fraud_rate_browser.index, y=fraud_rate_browser.values, palette='viridis')
    plt.title('Fraud Rate by Browser')
    plt.xlabel('Browser')
    plt.ylabel('Fraud Rate')
    plt.xticks(rotation=45)
    plt.show()
    
    # Fraud rate by source visualization - bar plot
    plt.figure(figsize=(12, 6))
    fraud_rate_source = df.groupby('source')['class'].mean().sort_values(ascending=False)
    sns.barplot(x=fraud_rate_source.index, y=fraud_rate_source.values, palette='magma')
    plt.title('Fraud Rate by Source')
    plt.xlabel('Source')
    plt.ylabel('Fraud Rate')
    plt.xticks(rotation=45)
    plt.show()
def preprocess_and_encode(merged_data):   
    # 1. **Normalization/Scaling** for numerical features
    scaler = MinMaxScaler()

    # List of numerical columns to normalize
    numerical_columns = ['purchase_value', 'age', 'time_diff_signup_purchase', 
                         'fraud_rate_by_browser', 'fraud_rate_by_source', 
                         'transaction_hour', 'transaction_day_of_week', 
                         'signup_year', 'purchase_year', 'signup_month', 
                         'purchase_month', 'signup_day', 'purchase_day', 
                         'signup_hour', 'purchase_hour']

    # Apply MinMax scaling (Normalization) to numerical columns
    merged_data[numerical_columns] = scaler.fit_transform(merged_data[numerical_columns])

    # 2. **Label Encoding** for ordinal categorical features (e.g., 'sex', 'source')
    label_encoder = LabelEncoder()

    # Columns with ordinal categorical data
    ordinal_columns = ['sex', 'source']

    for col in ordinal_columns:
        merged_data[col] = label_encoder.fit_transform(merged_data[col])

    # 3. **One-Hot Encoding** for nominal categorical features (e.g., 'browser', 'country')
    merged_data = pd.get_dummies(merged_data, columns=['browser', 'country'], drop_first=True)

    # 4. **Hashing for high-cardinality columns** (e.g., 'device_id')
    merged_data['device_id'] = merged_data['device_id'].apply(
        lambda x: int(hashlib.sha256(x.encode('utf-8')).hexdigest(), 16) % (10 ** 8)  # 8 digits hash
    )

    return merged_data