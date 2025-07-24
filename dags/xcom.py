from airflow.sdk import dag, task, Context
from typing import Dict, Any

@dag
def xcom_dag():

    @task
    def t1() -> Dict[str, Any]:
        my_val = 42
        my_sentence = "Hello, world!"
        return {
            "my_val": my_val,
            "my_sentence": my_sentence
        }

    @task
    def t2(data: Dict[str, Any]):
        print(data["my_val"])
        print(data["my_sentence"])

    val = t1()
    t2(val)

xcom_dag()