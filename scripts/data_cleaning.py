# data_clean.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipaddress
from sklearn.preprocessing import StandardScaler
import logging
def handle_missing_values(df):
    """
    Display missing value counts and drop rows missing critical values.
    """
    logging.info("Missing values before cleaning:")
    logging.info(df.isnull().sum())
    df = df.dropna(subset=['user_id', 'purchase_value', 'purchase_time'])
    return df

def remove_duplicates_and_convert_types(df):
    """
    Remove duplicate rows and convert time columns to datetime.
    """
    logging.info("Removing duplicates and converting time columns to datetime...")
    df = df.drop_duplicates()
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df
def univariate_analysis(df):
    """
    Performs univariate analysis on numerical and categorical features.
    
    Args:
        df (pd.DataFrame): The fraud dataset.
    
    Returns:
        None
    """
    logging.info("Performing univariate analysis...")
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Plot histograms and boxplots for numerical features
    for col in numerical_cols:
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Histogram of {col}')
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.show()
    
    # Plot count plots for categorical features
    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df[col], order=df[col].value_counts().index)
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=45)
        plt.show()


def bivariate_analysis(fraud_df):
    """
    Performs bivariate analysis on the fraud dataset.

    Args:
        fraud_df (pd.DataFrame): The cleaned fraud dataset.
    """
    print(fraud_df.columns)
    logging.info(fraud_df.columns)
    logging.info("Performing bivariate analysis...")
    # 1. Purchase Value vs. Age
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='purchase_value', data=fraud_df)
    plt.title('Purchase Value vs. Age')
    plt.xlabel('Age')
    plt.ylabel('Purchase Value ($)')
    plt.show()

    # 2. Purchase Value vs. Transaction Count
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='transaction_count', y='purchase_value', data=fraud_df)
    plt.title('Purchase Value vs. Transaction Count')
    plt.xlabel('Transaction Count')
    plt.ylabel('Purchase Value ($)')
    plt.show()

    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(fraud_df[['purchase_value', 'age', 'transaction_count']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


def ip_to_int(ip):
    """
    Converts an IP address string to an integer.
    """
    try:
        logging.info(f"Converting IP address {ip} to integer...")
        return int(ipaddress.ip_address(ip))
    except Exception:
        logging.error(f"Invalid IP address: {ip}")
        return np.nan

def merge_geolocation(fraud_df, geo_df):
    """
    Converts IP addresses to integer format and merges the country info into fraud_df.
    """
    logging.info("Merging geolocation data...")
    # Convert IP addresses to integer
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    
    # Ensure the geo dataset has integer bounds
    geo_df['lower_bound_ip_address'] = geo_df['lower_bound_ip_address'].astype(np.int64)
    geo_df['upper_bound_ip_address'] = geo_df['upper_bound_ip_address'].astype(np.int64)
    
    def map_ip_to_country(ip_int, geo_df):
        logging.info(f"Mapping IP address {ip_int} to country...")
        row = geo_df[(geo_df['lower_bound_ip_address'] <= ip_int) & 
                     (geo_df['upper_bound_ip_address'] >= ip_int)]
        if not row.empty:
            return row.iloc[0]['country']
        return np.nan

    fraud_df['country'] = fraud_df['ip_int'].apply(lambda x: map_ip_to_country(x, geo_df) if pd.notnull(x) else np.nan)
    return fraud_df

def feature_engineering(fraud_df):
    """
    Creates time-based features, calculates transaction frequency per user,
    normalizes continuous features, and encodes categorical features.
    """
    logging.info("Performing feature engineering...")
    # Time-based features
    fraud_df['purchase_hour'] = fraud_df['purchase_time'].dt.hour
    fraud_df['purchase_day_of_week'] = fraud_df['purchase_time'].dt.dayofweek

    # Transaction frequency per user (velocity)
    user_tx_counts = fraud_df.groupby('user_id').size().reset_index(name='transaction_count')
    fraud_df = fraud_df.merge(user_tx_counts, on='user_id', how='left')
    
    logging.info(f"normalized continuous features...")
    
    # Normalize continuous features
    scaler = StandardScaler()
    fraud_df[['purchase_value_scaled', 'age_scaled', 'transaction_count_scaled']] = scaler.fit_transform(
        fraud_df[['purchase_value', 'age', 'transaction_count']]
    )

    logging.info("Data hot encoding...")
    # One-hot encode categorical features (drop first to avoid dummy trap)
    fraud_df = pd.get_dummies(fraud_df, columns=['source', 'browser', 'sex', 'country'], drop_first=True)
    return fraud_df
