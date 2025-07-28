#!/usr/bin/env python
# coding: utf-8

"""
This script defines an Airflow DAG for orchestrating the training of a simple machine learning model to predict NYC taxi trip durations.
It includes tasks for setting up the environment, loading and preparing data, feature engineering, training the model and saving it using MLflow for experiment tracking and model registry.

Based on the orchestrated-duration-prediction-dag.py script, adapted for the 3rd homework questions (different training, model save and registry) 
"""

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta #to calculate relative dates
import pickle
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable


# Default arguments for the DAG
default_args = {
    'owner': 'andre',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'nyc_taxi_linear_duration_prediction',
    default_args=default_args,
    description='Train ML model to predict NYC taxi trip duration',
    schedule='@monthly',  # Run monthly
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'linear_regression', 'mlflow', 'taxi'],
)

#Global mlflow settings
mlflow.set_tracking_uri("http://03-orchestration-mlflow-1:5000") #docker container here
mlflow.set_experiment("nyc-taxi-experiment")
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def get_training_dates(**context):
    """Calculate training and validation dates based on current execution date"""
    logical_date = context['logical_date']
    
    # Training data: 4 months ago, because 2 months ago isn't available yet
    train_date = logical_date - relativedelta(months=4)
    train_year = train_date.year
    train_month = train_date.month
    
    # Validation data: 3 months ago  
    val_date = logical_date - relativedelta(months=3)
    val_year = val_date.year
    val_month = val_date.month
    
    print(f"Execution date: {logical_date.strftime('%Y-%m')}")
    print(f"Training data: {train_year}-{train_month:02d}")
    print(f"Validation data: {val_year}-{val_month:02d}")
    
    return {
        'train_year': train_year,
        'train_month': train_month,
        'val_year': val_year,
        'val_month': val_month
    }

def read_dataframe(year, month):
    """Read and preprocess taxi data"""
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    print(url)
    df = pd.read_parquet(url)
    original_shape = df.shape
    print(f"Original data shape: {df.shape}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    #As instructed: Use pick up and drop off locations separately, don't create a combination feature.
    #df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df, original_shape


def load_and_prepare_data(**context):
    """Load training and validation data"""
    # Get dates from previous task or calculate dynamically
    task_instance = context['task_instance']
    dates = task_instance.xcom_pull(task_ids='calculate_dates')

    train_year = dates['train_year']
    train_month = dates['train_month']
    val_year = dates['val_year']
    val_month = dates['val_month']
    
    print(f"Loading training data for: {train_year}-{train_month:02d}")
    print(f"Loading validation data for: {val_year}-{val_month:02d}")
    
    # Load training and validation data
    df_train, original_train_shape = read_dataframe(year=train_year, month=train_month)
    df_val, original_val_shape = read_dataframe(year=val_year, month=val_month)
    
    # Save dataframes for next tasks
    df_train.to_parquet('/tmp/df_train.parquet')
    df_val.to_parquet('/tmp/df_val.parquet')
    
    print(f"Training data shape: {df_train.shape}, from raw {original_train_shape} ")
    print(f"Validation data shape: {df_val.shape}, from raw {original_val_shape} ")
    
    return {"train_shape": df_train.shape, "val_shape": df_val.shape}


def create_X(df, dv=None):
    """Create feature matrix"""
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def prepare_features(**context):
    """Prepare features for training"""
    # Load dataframes
    df_train = pd.read_parquet('/tmp/df_train.parquet')
    df_val = pd.read_parquet('/tmp/df_val.parquet')
    
    # Create feature matrices
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)
    
    # Prepare target variables
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    
    # Save preprocessed data
    with open('/tmp/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('/tmp/X_val.pkl', 'wb') as f:
        pickle.dump(X_val, f)
    with open('/tmp/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('/tmp/y_val.pkl', 'wb') as f:
        pickle.dump(y_val, f)
    with open('/tmp/dv.pkl', 'wb') as f:
        pickle.dump(dv, f)
    
    print(f"Feature preparation completed")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")


def train_model(**context):
    """Train Linear Regression model with MLflow tracking"""
    # Load preprocessed data
    with open('/tmp/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('/tmp/X_val.pkl', 'rb') as f:
        X_val = pickle.load(f)
    with open('/tmp/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('/tmp/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)
    with open('/tmp/dv.pkl', 'rb') as f:
        dv = pickle.load(f)
    
    with mlflow.start_run() as run:
        # Train Linear Regression model with default parameters
        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_pred = linear_reg.predict(X_val)
        
        # Calculate RMSE on validation set
        rmse = root_mean_squared_error(y_val, y_pred)
        print(f"RMSE: {rmse:.2f} minutes")
        
        # Print the intercept of the model (required for Homework 3 question 5)
        print(f"Model intercept: {linear_reg.intercept_}")
        
        # Log parameters (Linear Regression has no hyperparameters to tune)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID, DOLocationID, trip_distance")
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        
        # Save preprocessor
        models_folder = Path('models')
        models_folder.mkdir(exist_ok=True)
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        
        # Get dates from previous task, for registering the training data version in the model name
        task_instance = context['task_instance']
        dates = task_instance.xcom_pull(task_ids='calculate_dates')
        year = dates['train_year']
        month = dates['train_month']   

        # Log model using MLflow
        mlflow.sklearn.log_model(
            linear_reg, 
            artifact_path="models_mlflow",
            registered_model_name=f"nyc-taxi-linear-regression-{year}-{month:02d}"
        )
        
        run_id = run.info.run_id

        # Check model size by downloading artifacts and reading MLmodel file
        # Optional. This can be seen in MLFlow UI, but if you want to print it in the logs you can uncomment this:
        """
        import os
        client = mlflow.tracking.MlflowClient()
        artifact_path = client.download_artifacts(run_id, "models_mlflow")
        mlmodel_path = os.path.join(artifact_path, "MLmodel")
        
        if os.path.exists(mlmodel_path):
            with open(mlmodel_path, 'r') as f:
                content = f.read()
                # Look for model_size_bytes in the MLmodel file
                for line in content.split('\n'):
                    if 'model_size_bytes' in line:
                        print(f"Model size: {line}")
                        break
                else:
                    # If model_size_bytes not found, calculate total size
                    total_size = sum(os.path.getsize(os.path.join(root, file)) 
                                   for root, dirs, files in os.walk(artifact_path) 
                                   for file in files)
                    print(f"model_size_bytes: {total_size}")
        """

        # Save run_id for potential downstream tasks
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        
        print(f"MLflow run_id: {run_id}")
        print(f"RMSE: {rmse}")
        
        return run_id


# Define Airflow tasks

calculate_dates_task = PythonOperator(
    task_id='calculate_dates',
    python_callable=get_training_dates,
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_and_prepare_data',
    python_callable=load_and_prepare_data,
    dag=dag,
)

prepare_features_task = PythonOperator(
    task_id='prepare_features',
    python_callable=prepare_features,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# Optional but recommended: Clean up temporary files
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command='rm -f /tmp/df_train.parquet /tmp/df_val.parquet /tmp/X_*.pkl /tmp/y_*.pkl /tmp/dv.pkl',
    dag=dag,
)

# Define task dependencies
calculate_dates_task >> load_data_task >> prepare_features_task >> train_model_task >> cleanup_task