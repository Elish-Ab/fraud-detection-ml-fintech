import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from flask import Flask, jsonify
# Import cleaning functions from your data_cleaning script
from scripts.data_cleaning import (
    load_data, 
    handle_missing_values, 
    remove_duplicates, 
    correct_data_types, 
    feature_engineering
)

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
# Load the raw data
df = load_data('fraud_data.csv')

# Clean the data using the provided functions
df = handle_missing_values(df)
df = remove_duplicates(df)
df = correct_data_types(df)
df = feature_engineering(df)

# For additional visualizations, compute extra features if not already present
if 'time_diff_signup_purchase' not in df.columns:
    df['time_diff_signup_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
if 'transaction_hour' not in df.columns:
    df['transaction_hour'] = df['purchase_time'].dt.hour
if 'is_night_transaction' not in df.columns:
    df['is_night_transaction'] = df['transaction_hour'].apply(lambda x: 1 if 0 <= x < 6 else 0)
if 'day_of_week' not in df.columns:
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

# Calculate summary statistics
total_transactions = len(df)
fraud_cases = df['class'].sum()
fraud_percentage = (fraud_cases / total_transactions) * 100 if total_transactions > 0 else 0

# -----------------------------
# Create Interactive Visualizations Using Plotly
# -----------------------------

# 1. Fraud (1) vs Non-Fraud (0) Transactions Count
fig_count = px.histogram(
    df, x="class", 
    title="Fraud (1) vs Non-Fraud (0) Transactions", 
    category_orders={'class': [0, 1]},
    labels={'class': 'Transaction Type'}
)

# 2. Boxplot: Transaction Value Distribution by Fraud Status
fig_box = px.box(
    df, x="class", y="purchase_value", 
    title="Transaction Value Distribution by Fraud Status",
    labels={'class': 'Fraud Status', 'purchase_value': 'Purchase Value'}
)

# 3. Histogram: Distribution of Purchase Value
fig_hist = px.histogram(
    df, x="purchase_value", nbins=50, 
    title="Distribution of Purchase Value",
    labels={'purchase_value': 'Purchase Value'}
)

# 4. Line Chart: Fraud Rate by Hour of the Day
fraud_rate_hour = df.groupby('transaction_hour')['class'].mean().reset_index(name='fraud_rate')
fig_hour = px.line(
    fraud_rate_hour, x='transaction_hour', y='fraud_rate', 
    title="Fraud Rate by Hour of the Day",
    labels={'transaction_hour': 'Hour of the Day', 'fraud_rate': 'Fraud Rate'}
)

# 5. Bar Chart: Fraud Rate by Day of the Week
fraud_rate_day = df.groupby('day_of_week')['class'].mean().reset_index(name='fraud_rate')
fig_day = px.bar(
    fraud_rate_day, x='day_of_week', y='fraud_rate', 
    title="Fraud Rate by Day of the Week",
    labels={'day_of_week': 'Day of the Week (0=Monday)', 'fraud_rate': 'Fraud Rate'}
)

# 6. Histogram: Time Difference (Signup to Purchase) in Hours
fig_time_diff = px.histogram(
    df, x="time_diff_signup_purchase", nbins=50, 
    title="Time Difference (Signup to Purchase) in Hours",
    labels={'time_diff_signup_purchase': 'Hours'}
)

# 7. Boxplot: Time Difference by Fraud Status
fig_time_box = px.box(
    df, x="class", y="time_diff_signup_purchase", 
    title="Time Difference (Signup to Purchase) by Fraud Status",
    labels={'class': 'Fraud Status', 'time_diff_signup_purchase': 'Hours'}
)

# 8. Count Plot: Night vs. Day Transactions
fig_night = px.histogram(
    df, x="is_night_transaction", 
    title="Count of Night vs. Day Transactions",
    category_orders={"is_night_transaction": [0, 1]},
    labels={'is_night_transaction': 'Is Night Transaction (1=Yes, 0=No)'}
)

# 9. Bar Chart: Fraud Rate by Browser (if 'browser' column exists)
if 'browser' in df.columns:
    fraud_rate_browser = df.groupby('browser')['class'].mean().reset_index(name='fraud_rate')
    fig_browser = px.bar(
        fraud_rate_browser, x='browser', y='fraud_rate', 
        title="Fraud Rate by Browser",
        labels={'browser': 'Browser', 'fraud_rate': 'Fraud Rate'}
    )
else:
    fig_browser = None

# 10. Bar Chart: Fraud Rate by Source (if 'source' column exists)
if 'source' in df.columns:
    fraud_rate_source = df.groupby('source')['class'].mean().reset_index(name='fraud_rate')
    fig_source = px.bar(
        fraud_rate_source, x='source', y='fraud_rate', 
        title="Fraud Rate by Source",
        labels={'source': 'Source', 'fraud_rate': 'Fraud Rate'}
    )
else:
    fig_source = None

# -----------------------------
# Set Up the Flask Server and API Endpoint
# -----------------------------

server = Flask(__name__)

@server.route('/fraud_summary', methods=['GET'])
def fraud_summary():
    summary = {
        'total_transactions': total_transactions,
        'fraud_cases': fraud_cases,
        'fraud_percentage': fraud_percentage
    }
    return jsonify(summary)

# -----------------------------
# Create the Dash Application
# -----------------------------

app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    
    # Summary Statistics
    html.Div([
        html.Div([
            html.H3("Total Transactions"),
            html.P(f"{total_transactions}")
        ], style={'padding': '10px', 'border': '1px solid #ccc', 'width': '30%', 'textAlign': 'center'}),
        html.Div([
            html.H3("Fraud Cases"),
            html.P(f"{fraud_cases}")
        ], style={'padding': '10px', 'border': '1px solid #ccc', 'width': '30%', 'textAlign': 'center'}),
        html.Div([
            html.H3("Fraud Percentage"),
            html.P(f"{fraud_percentage:.2f}%")
        ], style={'padding': '10px', 'border': '1px solid #ccc', 'width': '30%', 'textAlign': 'center'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
    
    # Graphs Section
    html.Div([
        dcc.Graph(figure=fig_count),
        dcc.Graph(figure=fig_box),
        dcc.Graph(figure=fig_hist),
        dcc.Graph(figure=fig_hour),
        dcc.Graph(figure=fig_day),
        dcc.Graph(figure=fig_time_diff),
        dcc.Graph(figure=fig_time_box),
        dcc.Graph(figure=fig_night),
        dcc.Graph(figure=fig_browser) if fig_browser else html.Div(),
        dcc.Graph(figure=fig_source) if fig_source else html.Div()
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
