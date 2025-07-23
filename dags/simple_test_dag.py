from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="my_example_dag",
    start_date=datetime(2025, 6, 1),
    schedule="@daily"
) as dag:
    task = BashOperator(
        task_id="print_hello",
        bash_command="echo Hello World"
    )