import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
from mlflow import MlflowClient

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
        self.model_name = "propension_abandono"
        self.model_version = -1

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

    def model_and_productivize(self):
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment("/Users/edufer01@ucm.es/experiment")
        mlflow.sklearn.autolog()
        with mlflow.start_run() as run:
            X_train = self.df_train[self.feature_cols]
            y_train = self.df_train['y']
            X_test = self.df_test[self.feature_cols]
            y_test = self.df_test['y']
            self.model = LogisticRegression(random_state=self.seed)
            self.model.fit(X_train, y_train)
            self.score = self.model.score(X_test, y_test)
            print(f"Score del modelo: {self.score:.2f}")

            y_pred = self.model.predict(X_test)
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="sklearn-model",
                signature=signature,
                registered_model_name=self.model_name,
            )

        client = MlflowClient()
        model_version = client.get_latest_versions(self.model_name, stages=["None"])[-1]
        self.model_version = int(model_version.__getattribute__('version'))
        print(f"Creada la version {self.model_version} de {self.model_name}")

    def save_train_test_metric_tables(self):

        self.df_metrics = pd.DataFrame([self.score], columns=['score'])

        self.df_train['model_name'] = self.model_name
        self.df_test['model_name'] = self.model_name
        self.df_metrics['model_name'] = self.model_name
        self.df_train['model_version'] = self.model_version
        self.df_test['model_version'] = self.model_version
        self.df_metrics['model_version'] = self.model_version

        sdf = self.spark.createDataFrame(self.df_train)
        sdf.write.mode("append").saveAsTable(self.table_train)
        sdf = self.spark.createDataFrame(self.df_test)
        sdf.write.mode("append").saveAsTable(self.table_test)
        sdf = self.spark.createDataFrame(self.df_metrics)
        sdf.write.mode("append").saveAsTable(self.table_metrics)


    def make_predictions(self):


        client = MlflowClient()
        self.model_version = client.get_latest_versions(self.model_name, stages=["None"])[-1]
        self.model = mlflow.sklearn.load_model(model_uri=f"models:/{self.model_name}/{self.model_version}")
        pass