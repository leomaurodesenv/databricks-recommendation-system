# Databricks notebook source
# MAGIC %md
# MAGIC # Data Science Modeling
# MAGIC
# MAGIC Data science modeling is a crucial step in the data science lifecycle that involves building predictive models using statistical and machine learning techniques. In this tutorial, we employ [Surprise](https://github.com/NicolasHug/Surprise) Data Science package to build our model. Susprise is a Python scikit for building and analyzing Recommendation Systems. 

# COMMAND ----------

# MAGIC %pip install scikit-surprise==1.1.3

# COMMAND ----------

import mlflow
import pandas as pd

from databricks import feature_store
from surprise import SVD, accuracy
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Feature Tables
# MAGIC
# MAGIC Databricks is a powerful data engineering and analytics platform that provides a wide range of tools for data processing, analysis, and visualization. One of the key features of Databricks is the ability to read feature tables, which are essentially structured data tables that contain a set of features or attributes that describe a particular entity or object. Feature tables are commonly used in machine learning and data science applications for tasks such as feature engineering, model training, and prediction.

# COMMAND ----------

# Create Feature Store client
fs = feature_store.FeatureStoreClient()

# Read Ratings from Feature Store
ratings = fs.read_table("default.ratings")
ratings.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validating and Testing
# MAGIC
# MAGIC Steps:
# MAGIC - Surprise has a set of builtin datasets, but you can create a custom dataset. Loading a rating dataset can be done either from a file (e.g. a csv file), or from a `pandas.DataFrame`. Either way, you will need to define a `Reader` object for Surprise to be able to parse the dataset.
# MAGIC - Run a cross validation procedure for a given algorithm, reporting accuracy measures and computation times.

# COMMAND ----------

# It only operates with pandas.DataFrame
ratings_df = ratings.toPandas()
ratings_range = ratings_df["Book-Rating"].min(), ratings_df["Book-Rating"].max()

# Creating surprise.Dataset
# The columns must correspond to user id, item id and ratings (in that order).
reader = Reader(rating_scale=(ratings_range[0], ratings_range[1]))
dataset = Dataset.load_from_df(ratings_df[["User-ID", "ISBN", "Book-Rating"]], reader)

# COMMAND ----------

# We'll use the famous SVD algorithm.
algorithm = SVD()

# Run 5-fold cross-validation and print results
results = cross_validate(algorithm, dataset, measures=["RMSE", "MAE"], cv=5, verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registering Experiment
# MAGIC MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. One of the key features of MLflow is the ability to register and track experiments, which allows data scientists and machine learning practitioners to keep track of their experiments and compare the performance of different models.
# MAGIC
# MAGIC Steps:
# MAGIC - Create an experiment
# MAGIC - Train SVD model
# MAGIC - Log trained model and results on MLflow

# COMMAND ----------

# Create an experiment name, which must be unique and case sensitive
experiment = mlflow.set_experiment("/Shared/book-recommendation-system")

print(f"\nName: {experiment.name}")
print(f"Experiment_id: {experiment.experiment_id}")
print(f"Tags: {experiment.tags}")
print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
print(f"Creation timestamp: {experiment.creation_time}")

# COMMAND ----------

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # train
    algorithm = SVD()
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=27)
    algorithm.fit(trainset)
    # score
    predictions = algorithm.test(testset)
    rmse = accuracy.rmse(predictions,verbose=False)
    mae = accuracy.mae(predictions,verbose=False)

    # logging experiment
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    # track model (depends on model type)
    mlflow.sklearn.log_model(algorithm, 'model')

f'RMSE: {rmse}', f'MAE: {mae}'
