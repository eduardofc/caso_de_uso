import numpy as np
import pandas as pd
import random


########################################################################################################################
# ESTE CÓDIGO ES PARA EJECUTAR EN EL NOTEBOOK
########################################################################################################################


def daily_events(date, clientes):
    n = len(clientes)
    x1 = np.random.uniform(1.0, 20.0, size=n)
    x2 = np.random.uniform(-10.0, 10.0, size=n)
    x3 = np.random.uniform(-1.0, 9.0, size=n)
    y = np.random.choice([0, 1], size=n, p=[0.95, 0.05])  # imbalanceado
    fecha = [date] * n
    df = pd.DataFrame(zip(fecha, clientes, x1, x2, x3, y), columns=['fecha', 'id_cliente', 'x1', 'x2', 'x3', 'y'])
    clientes_out =  df[df.y == 1]['id_cliente'].to_list() # clientes que se van de la compañía
    return df, clientes_out


def daily_features(date, clientes):
    n = len(clientes)
    x1 = np.random.uniform(1.0, 20.0, size=n)
    x2 = np.random.uniform(-10.0, 10.0, size=n)
    x3 = np.random.uniform(-1.0, 9.0, size=n)
    fecha = [date] * n
    df = pd.DataFrame(zip(fecha, clientes, x1, x2, x3), columns=['fecha', 'id_cliente', 'x1', 'x2', 'x3'])
    return df


def carga_31_enero():

    """ generamos 1000 clientes en la bbdd"""
    n = 1000
    clientes = [f"{np.random.randint(10000, 99999)}{chr(np.random.randint(65, 90))}" for _ in range(n)]

    """ cargamos la tabla histórica de daily_events para los días 1 al 31 de enero """
    fechas_enero = pd.date_range(start="2020-01-01", end="2020-01-31")
    fechas_enero = fechas_enero.strftime("%Y-%m-%d").to_list()
    df_final = None
    for ff in fechas_enero:
        df, clientes_out = daily_events(ff, random.sample(clientes, 100))  # asumo que solo cambian 100
        df_final = pd.concat([df_final, df]) if df_final is not None else df
        clientes = [c for c in clientes if c not in clientes_out]
    sdf_final = spark.createDataFrame(df_final)
    sdf_final.write.mode("overwrite").saveAsTable("default.daily_events")

    """ cargamos la tabla de features de clientes para predecir el día 1 de febrero """
    df_features = daily_features(date="2020-02-01", clientes=clientes) # hago prediccion para todos los clientes
    sdf_features = spark.createDataFrame(df_features)
    sdf_features.write.mode("overwrite").saveAsTable("default.daily_features")

    # devolvemos los clientes que aún siguen en la compañía
    return clientes


def carga_1_febrero(clientes):

    """ cargamos la tabla de daily_events para el día 1 feb """
    df_daily_events, clientes_out = daily_events(date="2020-02-01", clientes=random.sample(clientes, 100))
    clientes = [c for c in clientes if c not in clientes_out]
    sdf_daily_events = spark.createDataFrame(df_daily_events)
    sdf_daily_events.write.mode("append").saveAsTable("default.daily_events")

    """ cargamos la tabla de features de clientes para predecir el día 2 de febrero """
    df_features = daily_features(date="2020-02-02", clientes=clientes)
    sdf_features = spark.createDataFrame(df_features)
    sdf_features.write.mode("append").saveAsTable("default.daily_features")

    return clientes
