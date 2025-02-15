import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash import Input, Output
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Import cleaning functions from your data_cleaning script
# from data_cleaning import (
#     load_data,
#     handle_missing_values,
#     remove_duplicates,
#     correct_data_types,
#     feature_engineering
# )


# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
# Load the raw data
df = pd.read_csv('../data/Fraud_Data.csv')

# Additional features
df['purchase_time'] = pd.to_datetime(df['purchase_time'])
df['signup_time'] = pd.to_datetime(df['signup_time'])
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

# Create fraud rate by hour and day
fraud_rate_hour = df.groupby('transaction_hour')['class'].mean().reset_index(name='fraud_rate')
fraud_rate_day = df.groupby('day_of_week')['class'].mean().reset_index(name='fraud_rate')

# Create visualizations
def create_visualizations(df):
    # Fraud Count
    fig_count = px.histogram(
        df, x="class", title="Fraud (1) vs Non-Fraud (0) Transactions",
        category_orders={'class': [0, 1]}, labels={'class': 'Transaction Type (0: Non-Fraud, 1: Fraud)'},
        color_discrete_sequence=["#2D93AD", "#F16529"]
    )
    fig_count.update_traces(marker_line_width=1.5, opacity=0.8)
    fig_count.update_xaxes(title_text='Transaction Type')

    # Boxplot: Transaction Value Distribution
    fig_box = px.box(
        df, x="class", y="purchase_value",
        title="Transaction Value Distribution by Fraud Status",
        color="class",
        color_discrete_sequence=["#3A6A47", "#D64D4C"],
        labels={'class': 'Fraud Status', 'purchase_value': 'Purchase Value'}
    )
    fig_box.update_xaxes(title_text='Fraud Status')

    # Histogram: Purchase Value
    fig_hist = px.histogram(
        df, x="purchase_value", nbins=50,
        title="Distribution of Purchase Value",
        color_discrete_sequence=["#4D79A1"]
    )

    # Fraud Rate by Hour
    fig_hour = px.line(
        fraud_rate_hour, x='transaction_hour', y='fraud_rate',
        title="Fraud Rate by Hour of the Day",
        markers=True, color_discrete_sequence=["#EB6C73"]
    )

    # Fraud Rate by Day of the Week
    fig_day = px.bar(
        fraud_rate_day, x='day_of_week', y='fraud_rate',
        title="Fraud Rate by Day of the Week",
        color_discrete_sequence=["#F5A623"]
    )

    # Pie Chart: Fraud vs Non-Fraud Transactions
    fig_pie = px.pie(
        df, names='class', title="Proportion of Fraud vs Non-Fraud Transactions",
        labels={'class': 'Transaction Type'},
        color_discrete_sequence=["#FF6B6B", "#3D6A70"]
    )

    # Heatmap: Fraud Activity by Day of Week vs Hour of Day
    fraud_activity = df.groupby(['day_of_week', 'transaction_hour']).size().reset_index(name='count')
    fig_heatmap = px.density_heatmap(
        fraud_activity, x='transaction_hour', y='day_of_week', z='count',
        title="Fraud Activity by Day of Week and Hour of Day",
        color_continuous_scale="Viridis"
    )

    return fig_count, fig_box, fig_hist, fig_hour, fig_day, fig_pie, fig_heatmap

fig_count, fig_box, fig_hist, fig_hour, fig_day, fig_pie, fig_heatmap = create_visualizations(df)

# -----------------------------
# Create Dash App
# -----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            # Sidebar
            dbc.Nav([
                dbc.NavLink("Dashboard", href="#", active=True),
                dbc.NavLink("Transactions Overview", href="#"),
                dbc.NavLink("Detailed Statistics", href="#"),
            ], vertical=True, pills=True, className="bg-dark text-light p-3", style={"border-radius": "10px"}),
        ], width=2),
        dbc.Col([
            # Welcome Section
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H2("Welcome to the Fraud Detection Dashboard", className="text-center text-primary font-weight-bold"),
                    html.P("Explore the patterns of fraudulent transactions with interactive charts and insights.", className="text-center text-muted")
                ])), width=12),
            ], style={'margin-bottom': '30px'}),
            
            # Stats Cards
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Total Transactions", className="card-title text-success"),
                    html.H4(f"{total_transactions:,}", className="text-dark")
                ])), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Fraud Cases", className="card-title text-danger"),
                    html.H4(f"{fraud_cases:,}", className="text-dark")
                ])), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Fraud Percentage", className="card-title text-warning"),
                    html.H4(f"{fraud_percentage:.2f}%", className="text-dark")
                ])), width=4),
            ], style={'margin-bottom': '30px'}),
            
            # Insights Section
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Highest Fraud Rate Hour", className="card-title text-primary"),
                    html.H4(f"{fraud_rate_hour['fraud_rate'].idxmax()} Hour: {fraud_rate_hour['fraud_rate'].max():.2f} Fraud Rate", className="text-dark")
                ])), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Highest Fraud Rate Day", className="card-title text-primary"),
                    html.H4(f"Day {fraud_rate_day['fraud_rate'].idxmax()}: {fraud_rate_day['fraud_rate'].max():.2f} Fraud Rate", className="text-dark")
                ])), width=4)
            ], style={'margin-bottom': '30px'}),
            
            # Visualizations
            dbc.Row([
                dbc.Col(dcc.Graph(id='fig_count', figure=fig_count), md=6),
                dbc.Col(dcc.Graph(id='fig_box', figure=fig_box), md=6)
            ], style={'margin-bottom': '30px'}),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='fig_hist', figure=fig_hist), md=6),
                dbc.Col(dcc.Graph(id='fig_hour', figure=fig_hour), md=6)
            ], style={'margin-bottom': '30px'}),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='fig_day', figure=fig_day), md=6),
                dbc.Col(dcc.Graph(id='fig_pie', figure=fig_pie), md=6)
            ], style={'margin-bottom': '30px'}),
            
            # Heatmap
            dbc.Row([
                dbc.Col(dcc.Graph(id='fig_heatmap', figure=fig_heatmap), width=12)
            ], style={'margin-bottom': '30px'})
        ], width=10)
    ]),
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)
