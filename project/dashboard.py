import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from flask import Flask, jsonify
import dash_bootstrap_components as dbc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Import cleaning functions from your data_cleaning script
from data_cleaning import (
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
df = load_data('../data/Fraud_Data.csv')

df = handle_missing_values(df)
df = remove_duplicates(df)
df = correct_data_types(df)
df = feature_engineering(df)

# Additional features
df['time_diff_signup_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
df['transaction_hour'] = df['purchase_time'].dt.hour
df['is_night_transaction'] = df['transaction_hour'].apply(lambda x: 1 if 0 <= x < 6 else 0)
df['day_of_week'] = df['purchase_time'].dt.dayofweek

# Summary statistics
total_transactions = len(df)
fraud_cases = df['class'].sum()
fraud_percentage = (fraud_cases / total_transactions) * 100 if total_transactions > 0 else 0

# -----------------------------
# Create Interactive Visualizations Using Plotly
# -----------------------------

def create_visualizations(df):
    # Fraud Count
    fig_count = px.histogram(
        df, x="class", title="Fraud (1) vs Non-Fraud (0) Transactions",
        category_orders={'class': [0, 1]}, labels={'class': 'Transaction Type'},
        color_discrete_sequence=["#FF6361", "#58508D"]
    )
    fig_count.update_traces(marker_line_width=1.5, opacity=0.8)

    # Boxplot: Transaction Value Distribution
    fig_box = px.box(
        df, x="class", y="purchase_value",
        title="Transaction Value Distribution by Fraud Status",
        color="class",
        color_discrete_sequence=["#003F5C", "#FFA600"],
        labels={'class': 'Fraud Status', 'purchase_value': 'Purchase Value'}
    )

    # Histogram: Purchase Value
    fig_hist = px.histogram(
        df, x="purchase_value", nbins=50,
        title="Distribution of Purchase Value",
        color_discrete_sequence=["#2F4B7C"]
    )

    # Fraud Rate by Hour
    fraud_rate_hour = df.groupby('transaction_hour')['class'].mean().reset_index(name='fraud_rate')
    fig_hour = px.line(
        fraud_rate_hour, x='transaction_hour', y='fraud_rate',
        title="Fraud Rate by Hour of the Day",
        markers=True, color_discrete_sequence=["#D45087"]
    )
    
    return fig_count, fig_box, fig_hist, fig_hour

fig_count, fig_box, fig_hist, fig_hour = create_visualizations(df)

# -----------------------------
# Set Up Flask Server and API
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
# Create Dash App
# -----------------------------
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Fraud Detection Dashboard", className="text-center text-primary mb-4"))
    ]),
    
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Total Transactions", className="card-title text-secondary"),
            html.H4(f"{total_transactions:,}", className="text-dark")
        ]), color="light")),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Fraud Cases", className="card-title text-danger"),
            html.H4(f"{fraud_cases:,}", className="text-dark")
        ]), color="light")),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Fraud Percentage", className="card-title text-warning"),
            html.H4(f"{fraud_percentage:.2f}%", className="text-dark")
        ]), color="light")),
    ]),
    
    html.Hr(),

    dbc.Row([
        dbc.Col(dcc.Graph(id='fig_count', figure=fig_count), md=6),
        dbc.Col(dcc.Graph(id='fig_box', figure=fig_box), md=6)
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='fig_hist', figure=fig_hist), md=6),
        dbc.Col(dcc.Graph(id='fig_hour', figure=fig_hour), md=6)
    ]),
], fluid=True)

if __name__ == '__main__':
    app.run(debug=True, port=8051)