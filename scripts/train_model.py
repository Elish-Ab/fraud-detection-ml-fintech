import mlflow
import mlflow.sklearn
import mlflow.keras
import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv1D, Flatten, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")

sys.path.append('../scripts') 

# Data Preparation function
def prepare_data(df, target_col):
    # Drop rows with missing values first
    df_clean = df.dropna()
    if df_clean.empty:
        raise ValueError("DataFrame is empty after dropping missing values.")
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    return acc, f1, roc

# Training scikit-learn models with MLflow
def train_sklearn_model(model, model_name, X_train, X_test, y_train, y_test, params):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        acc, f1, roc = evaluate_model(model, X_test, y_test)
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('f1_score', f1)
        if roc is not None:
            mlflow.log_metric('roc_auc', roc)
        mlflow.sklearn.log_model(model, model_name)
        print(f"{model_name}: Accuracy={acc}, F1={f1}, ROC AUC={roc}")
        return model

# Create Keras models (MLP, CNN, RNN, LSTM)
def create_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model(input_dim):
    model = Sequential()
    model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(input_dim):
    model = Sequential()
    model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train Keras models with MLflow
def train_keras_model(model, model_name, X_train, X_test, y_train, y_test, params, epochs=50, batch_size=32):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train,
                            validation_split=0.2,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping],
                            verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.keras.log_model(model, model_name)
        print(f"{model_name} (Keras): Accuracy={accuracy}")
        return model

# Model explainability using SHAP
def explain_model_with_shap(model, X_test):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test, plot_type="bar")
    shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X_test.iloc[0, :], matplotlib=True)
    shap.dependence_plot(X_test.columns[0], shap_values[1], X_test)

# Model explainability using LIME
def explain_model_with_lime(model, X_train, X_test, instance_index=0):
    feature_names = X_train.columns.tolist()
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode='classification'
    )
    exp = lime_explainer.explain_instance(X_test.iloc[instance_index].values,
                                          model.predict_proba, num_features=10)
    print(f"LIME explanation for test instance {instance_index}:")
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight:.4f}")
    exp.show_in_notebook(show_table=True, show_all=False)

# Main function to train models on both datasets
def train_models_on_datasets():
    # -------------------------
    # E-commerce Data Experiment
    # -------------------------
    ecommerce_df = pd.read_csv('../data/Cleaned_Fraud_Data.csv')
    ecommerce_df['signup_time'] = pd.to_datetime(ecommerce_df['signup_time'], errors='coerce')
    ecommerce_df['purchase_time'] = pd.to_datetime(ecommerce_df['purchase_time'], errors='coerce')
    encoder = LabelEncoder()
    ecommerce_df['device_id'] = encoder.fit_transform(ecommerce_df['device_id'])
    ecommerce_df.drop(columns=['signup_time', 'purchase_time'], inplace=True)
    
    # Filter out rows where country is 'United States'
    filtered_df = ecommerce_df[ecommerce_df['country'] != 'United States']
    if filtered_df.empty:
        print("Warning: Filtering out 'United States' resulted in an empty dataset. Using the full dataset instead.")
        filtered_df = ecommerce_df.copy()
    else:
        ecommerce_df = filtered_df
        df = pd.read_csv("../data/Cleaned_Fraud_Data.csv")

        # Remove unwanted characters (if any column contains unexpected characters)
        df = df.replace(r"[^a-zA-Z0-9.,\s@_-]", "", regex=True)

        # Drop duplicates if necessary
        df = df.drop_duplicates()

        # Save the cleaned dataset
        df.to_csv("Cleaned_Fraud_Data_Cleaned.csv", index=False)

    try:
        X_train, X_test, y_train, y_test = prepare_data(ecommerce_df, 'class')
    except ValueError as e:
        print(f"Error preparing e-commerce data: {e}")
        return

    # Train models for E-commerce Data
    print("Training models for E-commerce Data")
    train_sklearn_model(LogisticRegression(max_iter=1000),
                          "Logistic Regression",
                          X_train, X_test, y_train, y_test,
                          {'max_iter': 1000})
    train_sklearn_model(RandomForestClassifier(n_estimators=100),
                          "Random Forest",
                          X_train, X_test, y_train, y_test,
                          {'n_estimators': 100})
    train_keras_model(create_mlp_model(X_train.shape[1]),
                      "MLP Keras",
                      X_train.to_numpy(), X_test.to_numpy(),
                      y_train.to_numpy(), y_test.to_numpy(),
                      {'model': 'MLP Keras'})

    # -------------------------
    # Credit Card Data Experiment
    # -------------------------
    credit_df = pd.read_csv('../data/creditcard.csv')
    # Convert the 'Time' column if necessary; drop it afterward
    credit_df['Time'] = pd.to_datetime(credit_df['Time'], unit='s', errors='coerce')
    credit_df = credit_df.drop(columns=['Time'], errors='ignore')  # Drop the Time column (if present)
    try:
        X_train, X_test, y_train, y_test = prepare_data(credit_df, 'Class')
    except ValueError as e:
        print(f"Error preparing credit card data: {e}")
        return

    # Train models for Credit Card Data
    print("Training models for Credit Card Data")
    train_sklearn_model(LogisticRegression(max_iter=1000),
                          "Logistic Regression (Credit Card)",
                          X_train, X_test, y_train, y_test,
                          {'max_iter': 1000})
    train_sklearn_model(RandomForestClassifier(n_estimators=100),
                          "Random Forest (Credit Card)",
                          X_train, X_test, y_train, y_test,
                          {'n_estimators': 100})
    train_keras_model(create_mlp_model(X_train.shape[1]),
                      "MLP Keras (Credit Card)",
                      X_train.to_numpy(), X_test.to_numpy(),
                      y_train.to_numpy(), y_test.to_numpy(),
                      {'model': 'MLP Keras'})

# Run the model training when executed as a script
if __name__ == '__main__':
    train_models_on_datasets()
