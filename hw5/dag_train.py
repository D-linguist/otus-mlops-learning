from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.contrib.hooks.ssh_hook import SSHHook
from airflow.contrib.operators.sftp_operator import SFTPOperator
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.models import Variable
from airflow.operators.bash import BashOperator


ssh_hook = SSHHook(ssh_conn_id='data_proc', cmd_timeout=None)


with DAG(
    dag_id='Train',
    default_args = { 
        'owner': 'Andrei',
        'depends_on_past': False,
        'email_on_failure': False,
    },
    schedule_interval=None,
    start_date=datetime(2023, 12, 1),
    catchup=False,
    tags=['hw5'],
    max_active_runs=1,
    dagrun_timeout=timedelta(days=1),
) as dag:

    copy_script = SFTPOperator(
        task_id='copy_script',
        ssh_hook=ssh_hook,
        local_filepath=str(Path(__file__).parent.resolve().joinpath('train.py')),
        remote_filepath='/home/ubuntu/train.py',
        operation='put',
    )

    train = SSHOperator(
        task_id="train_model",
        ssh_hook=ssh_hook,
        command='spark-submit train.py',
    )

    copy_script >> train