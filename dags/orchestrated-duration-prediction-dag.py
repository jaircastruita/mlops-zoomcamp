
#!/usr/bin/env python
# coding: utf-8

"""
This script defines an Airflow DAG for orchestrating the training of a simple machine learning model to predict NYC taxi trip durations.
It includes tasks for setting up the environment, loading and preparing data, feature engineering, training the model with XGBoost, and validating the model performance using MLflow for experiment tracking.

Used Claude Sonnet 4 for first draft and cleaned and corrected until having a PoC 
"""

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta #to calculate relative dates
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
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
    'nyc_taxi_duration_prediction',
    default_args=default_args,
    description='Train ML model to predict NYC taxi trip duration',
    schedule='@monthly',  # Run monthly
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'xgboost', 'mlflow', 'taxi'],
)

#Trying to make these variables global
mlflow.set_tracking_uri("http://03-orchestration-mlflow-1:5000") #docker container here
mlflow.set_experiment("nyc-taxi-experiment")
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

#Maybe not needed, but keeping for clarity
def setup_environment(**context): 
    """Setup MLflow and create necessary directories"""
    #mlflow.set_tracking_uri("http://localhost:5000") #default non containerized 
    mlflow.set_tracking_uri("http://03-orchestration-mlflow-1:5000") #the docker container here
    mlflow.set_experiment("nyc-taxi-experiment")
    
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)
    
    print("Environment setup completed")


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

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df, original_shape


def get_training_dates(**context):
    """Calculate training and validation dates based on current execution date"""
    #logical_date = context.get('logical_date')#, datetime.now())
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


def load_and_prepare_data(**context):
    """Load training and validation data"""
    # Get dates from previous task or calculate dynamically
    task_instance = context['task_instance']
    dates = task_instance.xcom_pull(task_ids='calculate_dates')
    
    # Fallback to manual variables if needed
    if not dates:
        train_year = int(Variable.get("training_year", default_var=datetime.now().year))
        train_month = int(Variable.get("training_month", default_var=datetime.now().month))
        val_year = train_year if train_month < 12 else train_year + 1
        val_month = train_month + 1 if train_month < 12 else 1
    else:
        train_year = dates['train_year']
        train_month = dates['train_month']
        val_year = dates['val_year']
        val_month = dates['val_month']
    
    print(f"Loading training data for: {train_year}-{train_month:02d}")
    print(f"Loading validation data for: {val_year}-{val_month:02d}")
    
    # Load training and validation data
    #df_train,  = read_dataframe(year=train_year, month=train_month)
    #df_val = read_dataframe(year=val_year, month=val_month)
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
    categorical = ['PU_DO']
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
    """Train XGBoost model with MLflow tracking"""
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
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Save preprocessor
        models_folder = Path('models')
        models_folder.mkdir(exist_ok=True)
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        # Log model
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        run_id = run.info.run_id
        
        # Save run_id for potential downstream tasks
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        
        print(f"MLflow run_id: {run_id}")
        print(f"RMSE: {rmse}")
        
        return run_id


def validate_model(**context):
    """Validate the trained model performance"""
    with open("run_id.txt", "r") as f:
        run_id = f.read().strip()
    
    # You can add model validation logic here
    # For example, checking if RMSE is below a threshold
    print(f"Validating model with run_id: {run_id}")
    
    # Example validation logic
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    rmse = run.data.metrics.get('rmse')
    
    if rmse and rmse < 10.0:  # Example threshold
        print(f"Model validation passed. RMSE: {rmse}")
        return True
    else:
        raise ValueError(f"Model validation failed. RMSE: {rmse}")


# Define tasks
""" 
#Let's ignore this task for now, 
#seems to work better as variables defined in the script than inside a task.
setup_task = PythonOperator(
    task_id='setup_environment',
    python_callable=setup_environment,
    dag=dag,
)
"""

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
"""
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
"""
# Also not important at this point. 
# Leaving it so it can be used later if needed.
"""
validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)
"""

# Optional but recommended: Clean up temporary files
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command='rm -f /tmp/df_train.parquet /tmp/df_val.parquet /tmp/X_*.pkl /tmp/y_*.pkl /tmp/dv.pkl',
    dag=dag,
)

# Define task dependencies
calculate_dates_task >> load_data_task >> cleanup_task
#prepare_features_task >> train_model_task >>
