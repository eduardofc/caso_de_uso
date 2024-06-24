import numpy as np
import pandas as pd
import random


########################################################################################################################
# ESTE CÓDIGO ES PARA EJECUTAR EN EL NOTEBOOK
########################################################################################################################

def daily_events(date, clientes):
    # clientes = [f"{np.random.randint(10000, 99999)}{chr(np.random.randint(65, 90))}" for _ in range(500)]
    n = len(clientes)
    x1 = np.random.uniform(1.0, 20.0, size=n)
    x2 = np.random.uniform(-10.0, 10.0, size=n)
    x3 = np.random.uniform(-1.0, 9.0, size=n)
    y = np.random.choice([0, 1], size=n, p=[0.95, 0.05])  # imbalanceado
    fecha = [date] * n
    df = pd.DataFrame(zip(fecha, clientes, x1, x2, x3, y), columns=['fecha', 'id_cliente', 'x1', 'x2', 'x3', 'y'])
    clientes_out =  df[df.y == 1]['id_cliente'].to_list()
    return df, clientes_out


def carga_los_eventos_de_enero(clientes, table_name="default.daily_events"):
    fechas_enero = pd.date_range(start="2020-01-01", end="2020-01-31")
    fechas_enero = fechas_enero.strftime("%Y-%m-%d").to_list()
    df_final = None
    for ff in fechas_enero:
        df, clientes_out = daily_events(ff, random.sample(clientes, 100))
        df_final = pd.concat([df_final, df]) if df_final is not None else df
        clientes = [c for c in clientes if c not in clientes_out]

    sdf_final = spark.createDataFrame(df_final)
    sdf_final.write.mode("overwrite").saveAsTable(table_name)
    return clientes


def carga_features_un_dia(date, clientes, table_name="default.daily_features"):
    df, clientes_out = daily_events(date, random.sample(clientes, 100))
    clientes = [c for c in clientes if c not in clientes_out]
    print(f">> Se han ido {len(clientes_out)} clientes de la compañía")
    df.drop(columns=['y'], inplace=True)
    sdf_final = spark.createDataFrame(df)
    sdf_final.write.mode("overwrite").saveAsTable(table_name)
    return clientes
