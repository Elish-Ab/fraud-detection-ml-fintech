import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
def merge_datasets(fraud_df, ip_df):
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)
    merged_df = fraud_df.merge(ip_df, how='left', left_on='ip_address', right_on='lower_bound_ip_address')
    return merged_df

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

# Function to normalize and scale features
def normalize_scale_features(df, columns):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Function to encode categorical features
def encode_categorical_features(df, columns):
    return pd.get_dummies(df, columns=columns)
