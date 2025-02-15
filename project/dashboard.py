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
df['age_group'] = pd.cut(df['age'], bins=[18, 30, 40, 50, 60, 100], labels=['18-30', '31-40', '41-50', '51-60', '60+'])

# Summary statistics
total_transactions = len(df)
fraud_cases = df['class'].sum()
fraud_percentage = (fraud_cases / total_transactions) * 100 if total_transactions > 0 else 0
non_fraud_cases = total_transactions - fraud_cases
non_fraud_percentage = 100 - fraud_percentage

# -----------------------------
# Create Interactive Visualizations Using Plotly
# -----------------------------

def create_visualizations(df):
    # Fraud vs Non-Fraud Overview (Pie Chart)
    fig_pie = px.pie(
        df, names='class', title="Fraud vs Non-Fraud Transactions",
        labels={'class': 'Transaction Type (0: Non-Fraud, 1: Fraud)'},
        color_discrete_sequence=["#FF6361", "#003F5C"],
        hole=0.3  # Donut chart for better visual appeal
    )
    fig_pie.update_traces(textinfo='percent+label', pull=[0.1, 0])  # Add pull effect for highlighting
    fig_pie.update_layout(margin=dict(t=50, b=50, l=50, r=50))

    # Transaction Insights (Boxplot)
    fig_box = px.box(df, x="class", y="purchase_value", color="class",
                     title="Transaction Insights: Fraud vs Non-Fraud Transaction Value",
                     labels={"class": "Fraud Class", "purchase_value": "Transaction Value ($)"})
    fig_box.update_layout(
        xaxis_title="Fraud Class (0: Non-Fraud, 1: Fraud)",
        yaxis_title="Transaction Value ($)",
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # Hourly fraud distribution
    fig_hour = px.histogram(df, x="transaction_hour", color="class", barmode="group", 
                             title="Fraud Distribution by Hour",
                             labels={"transaction_hour": "Hour of Day", "class": "Fraud Class"})
    fig_hour.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Number of Transactions",
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # Fraud by Browser (Updated - Top N browsers with most fraud cases)
    fraud_by_browser = df.groupby('browser')['class'].sum().reset_index()
    fraud_by_browser = fraud_by_browser.sort_values(by='class', ascending=False).head(10)  # Top 10 browsers
    fig_browser = px.bar(fraud_by_browser, x='browser', y='class', 
                         title="Fraud Cases by Browser (Top 10)",
                         labels={"browser": "Browser", "class": "Number of Fraud Cases"},
                         color='class', color_continuous_scale='Viridis')
    fig_browser.update_layout(
        xaxis_title="Browser",
        yaxis_title="Number of Fraud Cases",
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # Fraud by Device (Updated - Top N devices with most fraud cases)
    fraud_by_device = df.groupby('device_id')['class'].sum().reset_index()
    fraud_by_device = fraud_by_device.sort_values(by='class', ascending=False).head(10)  # Top 10 devices
    fig_device = px.bar(fraud_by_device, x='device_id', y='class', 
                        title="Fraud Cases by Device (Top 10)",
                        labels={"device_id": "Device", "class": "Number of Fraud Cases"},
                        color='class', color_continuous_scale='Cividis')
    fig_device.update_layout(
        xaxis_title="Device",
        yaxis_title="Number of Fraud Cases",
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # Fraud Distribution by Day of Week
    fraud_by_day = df.groupby('day_of_week')['class'].sum().reset_index()
    fig_day = px.bar(fraud_by_day, x='day_of_week', y='class', 
                     title="Fraud Distribution by Day of Week",
                     labels={"day_of_week": "Day of Week", "class": "Number of Fraud Cases"})
    fig_day.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Number of Fraud Cases",
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # Fraud vs Age Group (Fraud cases across age groups)
    fraud_by_age_group = df.groupby('age_group')['class'].sum().reset_index()
    fig_age = px.bar(fraud_by_age_group, x='age_group', y='class', 
                     title="Fraud vs Age Group",
                     labels={"age_group": "Age Group", "class": "Number of Fraud Cases"})
    fig_age.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Number of Fraud Cases",
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return {
        'fraud_overview': fig_pie,
        'box': fig_box,
        'hour': fig_hour,
        'device_browser': fig_device,
        'browser': fig_browser,
        'day': fig_day,
        'age': fig_age
    }

visualizations = create_visualizations(df)

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
        dbc.Col(
            dbc.NavbarSimple(
                brand="Fraud Detection Dashboard",
                brand_href="#",
                color="primary",
                dark=True
            ), width=12
        )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Nav([
                dbc.NavLink("Overview", id="overview-link", active=True),
                dbc.NavLink("Transaction Insights", id="transaction-link"),
                dbc.NavLink("Fraud Analysis", id="fraud-analysis-link"),
                dbc.NavLink("Device & Source Analysis", id="device-source-link"),
                dbc.NavLink("Fraud by Browser", id="browser-link"),
                dbc.NavLink("Fraud by Day", id="day-link"),
                dbc.NavLink("Fraud by Age Group", id="age-link")
            ], vertical=True, pills=True, className="bg-light p-3"),
        ], width=2),
        dbc.Col([
            dbc.Row([
                dbc.Col(html.H3("Fraud Detection Dashboard Overview", className="text-center my-3 text-primary"), width=12),
            ]),
            # Summary Stats for Total Transactions, Fraud Cases, and Percentages
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Total Transactions", className="card-title"),
                    html.P(f"{total_transactions}", className="card-text")
                ])), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Fraud Cases", className="card-title"),
                    html.P(f"{fraud_cases}", className="card-text")
                ])), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Fraud Percentage", className="card-title"),
                    html.P(f"{fraud_percentage:.2f}%", className="card-text")
                ])), width=4),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="fraud-overview", figure=visualizations['fraud_overview']), width=12),
                html.P("Fraud vs Non-Fraud Overview\nThis pie chart displays the distribution of fraud and non-fraud transactions. It provides an overview of the dataset, showing the percentage of fraudulent and non-fraudulent transactions.", style={"margin-top": "30px"}),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="boxplot", figure=visualizations['box']), width=12),
                html.P("Transaction Insights\nThis boxplot visualizes the transaction values for both fraud and non-fraud cases, highlighting differences in transaction size between the two classes.", style={"margin-top": "30px"}),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="hourly-fraud", figure=visualizations['hour']), width=12),
                html.P("Hourly Fraud Distribution\nThis histogram shows the distribution of fraud transactions across different hours of the day, which can help identify time patterns in fraud activity.", style={"margin-top": "30px"}),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="fraud-by-device", figure=visualizations['device_browser']), width=12),
                html.P("Fraud by Device\nThis bar chart shows the top 10 devices associated with the most fraud cases. It highlights which devices are more likely to be used in fraudulent transactions.", style={"margin-top": "30px"}),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="fraud-by-browser", figure=visualizations['browser']), width=12),
                html.P("Fraud by Browser\nThis bar chart shows the top 10 browsers used in fraudulent transactions. It allows us to understand the relationship between browser choice and fraud occurrence.", style={"margin-top": "30px"}),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="fraud-by-day", figure=visualizations['day']), width=12),
                html.P("Fraud Distribution by Day of Week\nThis chart displays the number of fraud cases across different days of the week, helping to spot any day-specific trends in fraudulent activity.", style={"margin-top": "30px"}),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="fraud-by-age", figure=visualizations['age']), width=12),
                html.P("Fraud vs Age Group\nThis chart compares fraud cases across different age groups, helping us understand which age groups are more susceptible to fraudulent activities.", style={"margin-top": "30px"}),
            ]),
        ], width=10),
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True)
