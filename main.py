from databricks.connect import DatabricksSession as SparkSession
from databricks.sdk.core import Config
from datetime import datetime

from jobs.job_model_training import ModelTraining
from jobs.job_make_predictions import MakePredictions
from utils import read_yaml
from model.model import Model


def train_model(spark, ):

    model = Model(spark)
    model.read_daily_events()
    model.train_test_split()
    model.model_and_productivize()
    model.save_train_test_metric_tables()
    model.make_predictions()


if __name__ == "__main__":

    # 1) inicialización de spark
    spark_config = Config(profile="DEFAULT", cluster_id="0619-142520-13uoteoc")
    spark = SparkSession.builder.sdkConfig(spark_config).getOrCreate()

    # 2) yaml con todos los parámetros
    config = read_yaml()

    # 3) fecha de ejecución del proceso completo (first execution o daily execution)
    exec_time = datetime.now()
    exec_time = exec_time.strftime('%Y-%m-%dT%H:%M:%S')

    """ 1 de febrero >> first-execution """

    # job model-training
    job = ModelTraining(spark=spark, config=config, exec_time=exec_time)
    job.run()

    # job make-predictions
    job = MakePredictions(spark=spark, config=config, exec_time=exec_time)
    job.run()


    table_daily_events = config['table']['daily_events']


    # a) partimos de un dataset de enero ya esté en bbdd
    # clientes = [f"{np.random.randint(10000, 99999)}{chr(np.random.randint(65, 90))}" for _ in range(500)]
    # df_enero = carga_los_eventos_de_enero(spark, clientes, table_name=table_daily_events)

    # b) entrenamos un modelo
    train_model(spark)


    # c) hacemos las predicciones para el 1 de febrero (y_hat)

    """ 1 de febrero """
    # a) nos dan los valores reales del 1 de febrero (y)

    # b) estudiamos si hay drift (calculamos la metrica con y y_hat)
    # if new_accuracy < accuracy_model_del_registry - 0.2

    # c) si hay drift reentramos,

    # d) hacemos las predicciones para el 2 de febrero

    """ 2 de febrero """
    # ...
