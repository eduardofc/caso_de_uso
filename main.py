from databricks.connect import DatabricksSession as SparkSession
from databricks.sdk.core import Config
import numpy as np

from utils import carga_los_eventos_de_enero, read_yaml
from model.model import Model


def train_model(spark, ):

    model = Model(spark)
    model.read_daily_events()
    model.train_test_split()
    model.modeling()
    model.save_train_test_metric_tables()


if __name__ == "__main__":

    config = Config(profile="DEFAULT", cluster_id="0619-142520-13uoteoc")
    spark = SparkSession.builder.sdkConfig(config).getOrCreate()

    config = read_yaml()
    table_daily_events = config['table']['daily_events']

    """ 1 de febrero """
    # a) partimos de un dataset de enero ya esté en bbdd
    # clientes = [f"{np.random.randint(10000, 99999)}{chr(np.random.randint(65, 90))}" for _ in range(500)]
    # df_enero = carga_los_eventos_de_enero(spark, clientes, table_name=table_daily_events)

    # b) entrenamos un modelo
    train_model(spark)


    # c) hacemos las predicciones para el 1 de febrero (y_hat)

    """ 1 de febrero """
    # a) nos dan los valores reales del 1 de febrero (y)

    # b) estudiamos si hay drift (calculamos la metrica con y y_hat)

    # c) si hay drift reentramos,

    # d) hacemos las predicciones para el 2 de febrero

    """ 2 de febrero """
    # ...
