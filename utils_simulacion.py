import numpy as np
import pandas as pd
import random


########################################################################################################################
# ESTE CÓDIGO ES PARA EJECUTAR EN EL NOTEBOOK
########################################################################################################################


# def monthly_events(date, clientes):
#     n = len(clientes)
#     x1 = np.random.uniform(1.0, 20.0, size=n)
#     x2 = np.random.uniform(-10.0, 10.0, size=n)
#     x3 = np.random.uniform(-1.0, 9.0, size=n)
#     y = np.random.choice([0, 1], size=n, p=[0.95, 0.05])  # imbalanceado
#     fecha = [date] * n
#     df = pd.DataFrame(zip(fecha, clientes, x1, x2, x3, y), columns=['fecha', 'id_cliente', 'x1', 'x2', 'x3', 'y'])
#     clientes_out =  df[df.y == 1]['id_cliente'].to_list() # clientes que se van de la compañía
#     return df, clientes_out
#
# def monthly_features(date, clientes):
#     n = len(clientes)
#     x1 = np.random.uniform(1.0, 20.0, size=n)
#     x2 = np.random.uniform(-10.0, 10.0, size=n)
#     x3 = np.random.uniform(-1.0, 9.0, size=n)
#     fecha = [date] * n
#     df = pd.DataFrame(zip(fecha, clientes, x1, x2, x3), columns=['fecha', 'id_cliente', 'x1', 'x2', 'x3'])
#     return df
#
#
# def carga_1_febrero():
#
#     """ generamos 1000 clientes en la bbdd"""
#     n = 1000
#     clientes = [f"{np.random.randint(10000, 99999)}{chr(np.random.randint(65, 90))}" for _ in range(n)]
#
#     """ daily_events para los días 1 al 31 de enero (histórico) """
#     # con este dataset haremos el modelo
#     fechas_enero = pd.date_range(start="2020-01-01", end="2020-01-31")
#     fechas_enero = fechas_enero.strftime("%Y-%m-%d").to_list()
#     df_final = None
#     for ff in fechas_enero:
#         df, clientes_out = daily_events(ff, random.sample(clientes, 100))  # asumo que solo cambian 100
#         df_final = pd.concat([df_final, df]) if df_final is not None else df
#         clientes = [c for c in clientes if c not in clientes_out]
#     sdf_final = spark.createDataFrame(df_final)
#     sdf_final.write.mode("overwrite").saveAsTable("default.daily_events")
#
#     """ daily_features del día 1 de febrero """
#     # con este dataset prediciremos quién se irá el día 1 de febrero, estarán las features de los clientes que se van a ir
#     df_features = daily_features(date="2020-02-01", clientes=clientes) # hago prediccion para todos los clientes
#     sdf_features = spark.createDataFrame(df_features)
#     sdf_features.write.mode("overwrite").saveAsTable("default.daily_features")
#
#     # devolvemos los clientes que aún siguen en la compañía
#     return clientes
#
#
# def carga_2_febrero(clientes):
#
#     """ daily_events para el día 1 de febrero """
#     df_daily_events, clientes_out = daily_events(date="2020-02-01", clientes=random.sample(clientes, 100))
#     clientes = [c for c in clientes if c not in clientes_out]
#     sdf_daily_events = spark.createDataFrame(df_daily_events)
#     sdf_daily_events.write.mode("append").saveAsTable("default.daily_events")
#
#     """ daily_features del día 2 de febrero """
#     df_features = daily_features(date="2020-02-02", clientes=clientes)
#     sdf_features = spark.createDataFrame(df_features)
#     sdf_features.write.mode("append").saveAsTable("default.daily_features")
#
#     return clientes


def carga_31_enero():

    # generamos 1000 clientes en la bbdd, con sus códigos aleatorios a priori
    n = 1000
    print(f"Tenemos {n} clientes")

    id_clientes = list(range(1,n+1))
    mes = '01' * n
    x1 = np.random.uniform(1.0, 20.0, size=n)
    x2 = np.random.uniform(-10.0, 10.0, size=n)
    x3 = np.random.uniform(-1.0, 9.0, size=n)
    y = np.random.choice([0, 1], size=n, p=[0.95, 0.05])  # imbalanceado

    return pd.DataFrame(zip(id_clientes, mes, x1, x2, x3, y), columns=['id_cliente','mes','x1','x2','x3','y'])


def carga_1_febrero(df_events):

    # Ya no nos interesan los que no son clientes para calcular
    df_aux = df_events[df_events.y != 1]
    n = len(df_aux)
    print(f"Tenemos {n} clientes")
    id_clientes = df_aux.clientes.to_list()

    mes = '02' * n
    x1 = np.random.uniform(1.0, 20.0, size=n)
    x2 = np.random.uniform(-10.0, 10.0, size=n)
    x3 = np.random.uniform(-1.0, 9.0, size=n)

    return pd.DataFrame(zip(id_clientes, mes, x1, x2, x3), columns=['id_cliente','mes','x1','x2','x3'])

def carga_1_marzo(df_events, df_features):

    # df_events tiene que appendar la info de febrero
    # Para ello, reutilizaremos las features de febrero y la target la generaremos aleatoriamente
    df_aux = df_features[df_features.mes=='02'].copy()
    n = len(df_aux)
    df_aux['y'] = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    df_events = pd.concat([df_events, df_aux])

    # df_features debe generar valores nuevos para los clientes que queden a final de mes pasado
    df_aux = df_events[(df_events.mes=='02') & (df_events.y != 1)]
    n = len(df_aux)
    print(f"Tenemos {n} clientes")
    id_clientes = df_aux.clientes.to_list()

    mes = '03' * n
    x1 = np.random.uniform(1.0, 20.0, size=n)
    x2 = np.random.uniform(-10.0, 10.0, size=n)
    x3 = np.random.uniform(-1.0, 9.0, size=n)

    df_aux = pd.DataFrame(zip(id_clientes, mes, x1, x2, x3), columns=['id_cliente','mes','x1','x2','x3'])
    df_features = pd.concat([df_features, df_aux])

    return df_events, df_features


def carga_1_abril(df_events, df_features):
    pass


if __name__ == "__main__":

    # El día 31 de enero, nos piden hacer un modelo para poner en producción desde mañana
    # Para ello, nos dan la tabla monthly_events con la info del último mes
    df_monthly_events = carga_31_enero()  # DISEÑO DEL MODELO

    # El día 1 de febrero, nos piden productificar el modelo
    # Para ello, nos dan la tabla de monthly_features
    df_monthly_features = carga_1_febrero(df_monthly_events)  # PRODUCTIFICACIÓN DEL MODELO >> HACER PREDICCIONES

    # El día 1 de marzo, nos piden evaluar si el modelo está funcionando bien,
    # reentrenar si fuera necesario y volver a hacer predicciones
    # Para ello, actualizarán mes a mes, las tablas monthly_events y monthly_features
    df_monthly_events, df_monthly_features = carga_1_marzo(df_monthly_events, df_monthly_features)

    # Y así, mes a mes...
    # df_monthly_events, df_monthly_features = carga_1_abril(df_monthly_events, df_monthly_features)
