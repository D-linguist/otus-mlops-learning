from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.contrib.hooks.ssh_hook import SSHHook
from airflow.contrib.operators.sftp_operator import SFTPOperator
from airflow.providers.ssh.operators.ssh import SSHOperator


ssh_hook = SSHHook(ssh_conn_id='data_proc', cmd_timeout=None)


with DAG(
    dag_id='Validate',
    default_args = { 
        'owner': 'Andrei',
        'depends_on_past': False,
        'email_on_failure': False,
    },
    schedule_interval=None,
    start_date=datetime(2023, 12, 1),
    catchup=False,
    tags=['hw6'],
    max_active_runs=1,
    dagrun_timeout=timedelta(days=1),
) as dag:

    copy_script = SFTPOperator(
        task_id='copy_script',
        ssh_hook=ssh_hook,
        local_filepath=str(Path(__file__).parent.resolve().joinpath('validate.py')),
        remote_filepath='/home/ubuntu/validate.py',
        operation='put',
    )

    validate = SSHOperator(
        task_id="validate",
        ssh_hook=ssh_hook,
        command='spark-submit validate.py',
    )

    copy_script >> validate