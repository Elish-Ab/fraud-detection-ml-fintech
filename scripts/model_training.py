import mlflow
import mlflow.sklearn
import mlflow.keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from lime import lime_tabular
import shap

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
sys.path.append('../scripts') 
def prepare_data(df, target_col):
    """
    Splits the DataFrame into training and testing sets.
    Assumes the target column exists in the DataFrame.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns accuracy, F1-score, and ROC AUC (if available).
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    return acc, f1, roc

def train_sklearn_model(model, model_name, X_train, X_test, y_train, y_test, params):
    """
    Trains a scikit-learn model, logs parameters and metrics using MLflow.
    """
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

def train_keras_model(model, model_name, X_train, X_test, y_train, y_test, params, epochs=50, batch_size=32):
    """
    Trains a Keras model, logs parameters and metrics using MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.keras.log_model(model, model_name)
        print(f"{model_name} (Keras): Accuracy={accuracy}")
        return model

def explain_model_with_shap(model, X_test):
    """
    Explains the model predictions using SHAP for tree-based models.
    """
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values[1], X_test, plot_type="bar")
    shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X_test.iloc[0, :], matplotlib=True)
    shap.dependence_plot(X_test.columns[0], shap_values[1], X_test)

def explain_model_with_lime(model, X_train, X_test, instance_index=0):
    """
    Uses LIME to explain a model's prediction for a single instance.
    """
    feature_names = X_train.columns.tolist()
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode='classification'
    )
    
    exp = lime_explainer.explain_instance(
        X_test.iloc[instance_index].values,
        model.predict_proba,
        num_features=10
    )
    
    exp.show_in_notebook(show_table=True, show_all=False)

def train_all_models(df, target_col):
    """
    Prepares the data and trains all selected models for both datasets.
    """
    X_train, X_test, y_train, y_test = prepare_data(df, target_col)
    input_dim = X_train.shape[1]
    
    print("Training Scikit-Learn Models...")
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    train_sklearn_model(lr, "Logistic Regression", X_train, X_test, y_train, y_test, {'max_iter': 1000})
    
    # Decision Tree
    dt = DecisionTreeClassifier()
    train_sklearn_model(dt, "Decision Tree", X_train, X_test, y_train, y_test, {})
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    trained_rf = train_sklearn_model(rf, "Random Forest", X_train, X_test, y_train, y_test, {'n_estimators': 100})
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100)
    train_sklearn_model(gb, "Gradient Boosting", X_train, X_test, y_train, y_test, {'n_estimators': 100})
    
    # MLP using Scikit-Learn
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    train_sklearn_model(mlp, "MLP Sklearn", X_train, X_test, y_train, y_test, {'hidden_layer_sizes': (100,), 'max_iter': 300})
    
    print("Training Keras Models...")
    
    # Convert data to numpy arrays for Keras
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    
    # MLP using Keras
    mlp_keras = create_mlp_model_keras(input_dim)
    train_keras_model(mlp_keras, "MLP Keras", X_train_np, X_test_np, y_train_np, y_test_np, {'model': 'MLP Keras'})
    
    # CNN
    cnn_model = create_cnn_model(input_dim)
    train_keras_model(cnn_model, "CNN", X_train_np, X_test_np, y_train_np, y_test_np, {'model': 'CNN'})
    
    # RNN
    rnn_model = create_rnn_model(input_dim)
    train_keras_model(rnn_model, "RNN", X_train_np, X_test_np, y_train_np, y_test_np, {'model': 'RNN'})
    
    # LSTM
    lstm_model = create_lstm_model(input_dim)
    train_keras_model(lstm_model, "LSTM", X_train_np, X_test_np, y_train_np, y_test_np, {'model': 'LSTM'})
    
    # Return the Random Forest model and both X_train and X_test for explainability
    return trained_rf, X_train, X_test

# ----------------------
# Main Execution: Train on Both Datasets
# ----------------------
if __name__ == '__main__':
    # Load the datasets
    ecommerce_df = pd.read_csv('../data/Preprocessed_Data.csv')
    creditcard_df = pd.read_csv('../data/creditcard.csv')
    
    print("Training models for E-commerce Data")
    trained_rf_ecommerce, X_train_ecommerce, X_test_ecommerce = train_all_models(ecommerce_df, 'class')
    explain_model_with_shap(trained_rf_ecommerce, X_test_ecommerce)
    explain_model_with_lime(trained_rf_ecommerce, X_train_ecommerce, X_test_ecommerce)
    
    print("Training models for Credit Card Data")
    trained_rf_creditcard, X_train_creditcard, X_test_creditcard = train_all_models(creditcard_df, 'Class')
    explain_model_with_shap(trained_rf_creditcard,X_train_creditcard,X_test_creditcard)
