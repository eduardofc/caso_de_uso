import numpy as np
import pandas as pd
import random
import yaml

def daily_events(date, clientes):
    n = len(clientes)
    x1 = np.random.uniform(1.0, 20.0, size=n)
    x2 = np.random.uniform(-10.0, 10.0, size=n)
    x3 = np.random.uniform(-1.0, 9.0, size=n)
    y = np.random.choice([0, 1], size=n, p=[0.95, 0.05])  # imbalanceado
    fecha = [date] * n
    df = pd.DataFrame(zip(fecha, clientes, x1, x2, x3, y), columns=['fecha', 'id_cliente', 'x1', 'x2', 'x3', 'y'])
    clientes_out = df[df.y == 1]['id_cliente'].to_list()
    return df, clientes_out


def carga_los_eventos_de_enero(spark, clientes, table_name):
    fechas_enero = pd.date_range(start="2020-01-01", end="2020-01-31")
    fechas_enero = fechas_enero.strftime("%Y-%m-%d").to_list()
    df_final = None
    for ff in fechas_enero:
        df, clientes_out = daily_events(ff, random.sample(clientes, 100))
        df_final = pd.concat([df_final, df]) if df_final is not None else df

    sdf_final = spark.createDataFrame(df_final)
    sdf_final.write.mode("overwrite").saveAsTable(table_name)


def read_yaml():
    with open("config.yaml", "rb") as file:
        config = yaml.safe_load(file)
    return config