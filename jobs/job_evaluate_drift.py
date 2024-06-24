from datetime import datetime, timedelta
import pandas as pd
from sklearn.metrics import accuracy_score


class EvaluateDrift():

    def __init__(self, spark, config, exec_time, date=None):
        self.spark = spark
        self.exec_time = exec_time

        # restamos un día a la fecha actual
        if date is None:
            date = datetime.now()
        else:
            date = datetime.strptime(date, '%Y-%m-%d')
        date = date - timedelta(days=1)
        self.date = date.strftime('%Y-%m-%d')

        # config
        self.table_daily_predictions = config['table']['daily_predictions']
        self.table_daily_events = config['table']['daily_events']

    def run(self):

        # load y, y_hat
        df_y = (
            self.spark.read.table(self.table_daily_events)
            .where(f"fecha='{self.date}'")
            .toPandas()
        )
        df_y_hat = (
            self.spark.read.table(self.table_daily_predictions)
            .where(f"fecha='{self.date}'")
            .toPandas()
        )
        df = pd.merge(df_y, df_y_hat, on="id_cliente", how="inner")

        # compute score
        y = df.y.to_list()
        y_hat = df.y_hat.to_list()
        score = accuracy_score(y_true=y, y_pred=y_hat)
        print(f"Score del modelo para el día {self.date}: {score:.2f}")
