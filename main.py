from databricks.connect import DatabricksSession as SparkSession
from databricks.sdk.core import Config
from datetime import datetime

from jobs.job_evaluate_drift import EvaluateDrift
from jobs.job_model_training import ModelTraining
from jobs.job_make_predictions import MakePredictions
from utils import read_yaml


if __name__ == "__main__":

    # 1) inicialización de spark
    spark_config = Config(profile="DEFAULT", cluster_id="0619-142520-13uoteoc")
    spark = SparkSession.builder.sdkConfig(spark_config).getOrCreate()

    # 2) yaml con todos los parámetros
    config = read_yaml()

    # 3) fecha de ejecución del proceso completo (first execution o daily execution)
    exec_time = datetime.now()
    exec_time = exec_time.strftime('%Y-%m-%dT%H:%M:%S')

    """ 31 de enero >> first-execution """

    # job model-training
    job = ModelTraining(spark=spark, config=config, exec_time=exec_time)
    job.run()

    # job make-predictions
    job = MakePredictions(spark=spark, config=config, exec_time=exec_time, date="2020-02-01")
    job.run()

    """ 1 de febrero >> daily-execution """

    # job evaluate-drift
    job = EvaluateDrift(spark=spark, config=config, exec_time=exec_time, date="2020-02-02")
    job.run()


