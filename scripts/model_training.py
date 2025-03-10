import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Import libraries
import sys
import mlflow
import mlflow.sklearn
import mlflow.keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Sklearn and other ML libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# TensorFlow and Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, MaxPooling1D
from keras.layers import Conv1D, SimpleRNN, Flatten, Reshape
from keras.optimizers import Adam

# Lime and SHAP
from lime import lime_tabular
import shap

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
sys.path.append('../scripts')

# Functions
def prepare_data(df, target_col):
    """
    Splits the DataFrame into training and testing sets and handles class imbalance.
    Assumes the target column exists in the DataFrame.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
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

def train_sklearn_model_with_tuning(model, model_name, X_train, X_test, y_train, y_test, params):
    """
    Trains a scikit-learn model with hyperparameter tuning using GridSearchCV.
    Logs parameters and metrics using MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        
        # Hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        acc, f1, roc = evaluate_model(best_model, X_test, y_test)
        
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('f1_score', f1)
        if roc is not None:
            mlflow.log_metric('roc_auc', roc)
        
        mlflow.sklearn.log_model(best_model, model_name)
        print(f"{model_name}: Accuracy={acc}, F1={f1}, ROC AUC={roc}")
        return best_model

def create_mlp_model_keras(input_dim):
    """
    Creates an MLP model in Keras for classification.
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model(input_dim):
    """
    Creates a CNN model in Keras for classification.
    """
    model = Sequential()
    model.add(Conv1D(64, 2, activation='relu', input_shape=(input_dim, 1)))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model(input_dim):
    """
    Creates an LSTM model in Keras for classification.
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=(input_dim, 1), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(input_shape):
    """
    Create and return an RNN model.
    """
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_keras_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train the Keras model and log metrics.
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print(f"Training {model_name}...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop], verbose=2)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"{model_name} - Test Accuracy: {accuracy}, Test Loss: {loss}")
    
    # Log model evaluation metrics (you can use MLflow here or another logging method)
    return history

def explain_model_with_shap(model, X_test):
    """
    Explain model predictions using SHAP.
    Assumes binary classification and uses KernelExplainer.
    """
    explainer = shap.KernelExplainer(model.predict, X_test)
    shap_values = explainer.shap_values(X_test)
    
    # Visualize SHAP values with a summary plot
    shap.summary_plot(shap_values, X_test)
    
    # Optionally, visualize individual predictions
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0])
    plt.show()  # Ensure SHAP plot is shown

def train_all_models(df, target_col, use_smote=True):
    """
    Prepares the data and trains all selected models for both datasets with hyperparameter tuning.
    """
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df, target_col)
    input_dim = X_train.shape[1]
    
    print("Training Scikit-Learn Models...")
    
    # Logistic Regression with class weights and hyperparameter tuning
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_params = {'C': [0.01, 0.1, 1, 10]}
    train_sklearn_model_with_tuning(lr, "Logistic Regression", X_train, X_test, y_train, y_test, lr_params)
    
    # Decision Tree with hyperparameter tuning
    dt = DecisionTreeClassifier(class_weight='balanced')
    dt_params = {'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]}
    train_sklearn_model_with_tuning(dt, "Decision Tree", X_train, X_test, y_train, y_test, dt_params)
    
    # Random Forest with class weight and hyperparameter tuning
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100)
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
    trained_rf = train_sklearn_model_with_tuning(rf, "Random Forest", X_train, X_test, y_train, y_test, rf_params)

    # Gradient Boosting with hyperparameter tuning
    gb = GradientBoostingClassifier()
    gb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
    train_sklearn_model_with_tuning(gb, "Gradient Boosting", X_train, X_test, y_train, y_test, gb_params)

    print("Training Neural Networks (Keras)...")
    
    # MLP Model
    mlp_model = create_mlp_model_keras(input_dim)
    train_keras_model(mlp_model, "MLP Model", X_train, X_test, y_train, y_test)
    
    # CNN Model
    cnn_model = create_cnn_model(input_dim)
    train_keras_model(cnn_model, "CNN Model", X_train, X_test, y_train, y_test)
    
    # LSTM Model
    lstm_model = create_lstm_model(input_dim)
    train_keras_model(lstm_model, "LSTM Model", X_train, X_test, y_train, y_test)
    
    # RNN Model
    rnn_model = create_rnn_model(X_train.shape[1:])
    train_keras_model(rnn_model, "RNN Model", X_train, X_test, y_train, y_test)
    
    # Explain the best model (RNN in this case) using SHAP
    explain_model_with_shap(rnn_model, X_test)

# Main Execution (adjust this as per your dataset)
df = pd.read_csv("data/Preprocessed_Data.csv")

target_col = 'class'  
train_all_models(df, target_col)
