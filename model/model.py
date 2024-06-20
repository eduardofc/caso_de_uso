import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow

from utils import read_yaml


class Model():
    def __init__(self, spark):
        self.spark = spark
        self.feature_cols = ['x1', 'x2', 'x3']
        self.df_events = None
        self.df_train = None
        self.df_test = None
        self.df_metrics = None
        self.model = None
        self.score = -1

        config = read_yaml()
        self.table_daily_events = config['table']['daily_events']
        self.table_train = config['table']['train']
        self.table_test = config['table']['test']
        self.table_metrics = config['table']['metrics']
        self.seed = config['modeling']['seed']
        self.test_size = config['modeling']['test_size']

    def read_daily_events(self):
        self.df_events = self.spark.read.table(self.table_daily_events).toPandas()

    def train_test_split(self):
        X = self.df_events[self.feature_cols]
        y = self.df_events['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.seed)
        self.df_train = pd.merge(X_train, y_train, left_index=True, right_index=True, how='inner')
        self.df_test = pd.merge(X_test, y_test, left_index=True, right_index=True, how='inner')

    def modeling(self):
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment("/Users/edufer01@ucm.es/experiment")
        with mlflow.start_run():
            X_train = self.df_train[self.feature_cols]
            y_train = self.df_train['y']
            X_test = self.df_test[self.feature_cols]
            y_test = self.df_test['y']
            self.model = LogisticRegression(random_state=self.seed)
            self.model.fit(X_train, y_train)
            self.score = self.model.score(X_test, y_test)
            print(f"Score del modelo: {self.score:.2f}")

    def save_train_test_metric_tables(self):

        self.df_metrics = pd.DataFrame([self.score], columns=['score'])

        model_name = "propension_abandono"
        model_version = -1
        self.df_train['model_name'] = model_name
        self.df_test['model_name'] = model_name
        self.df_metrics['model_name'] = model_name
        self.df_train['model_version'] = model_version
        self.df_test['model_version'] = model_version
        self.df_metrics['model_version'] = model_version

        sdf = self.spark.createDataFrame(self.df_train)
        sdf.write.mode("append").saveAsTable(self.table_train)
        sdf = self.spark.createDataFrame(self.df_test)
        sdf.write.mode("append").saveAsTable(self.table_test)
        sdf = self.spark.createDataFrame(self.df_metrics)
        sdf.write.mode("append").saveAsTable(self.table_metrics)
