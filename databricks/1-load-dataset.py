# Databricks notebook source
import zipfile
import pandas as pd

from pathlib import Path
from databricks import feature_store

# COMMAND ----------

# MAGIC %md
# MAGIC # Download and Load Dataset
# MAGIC
# MAGIC To build a Recommendation System (RecSys), the first step is to gather a relevant dataset. There are various sources from where you can download datasets such as GitHub, Huggingface, or Kaggle. In this repository, we will employ the [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?select=Ratings.csv). Book Recommendation Dataset is a RecSys dataset about books, users and ratings.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Dataset
# MAGIC
# MAGIC To download the dataset, we will use the [Kaggle API](https://github.com/Kaggle/kaggle-api). You must fill the `kaggle.json` file in the root folder of this project.
# MAGIC
# MAGIC Steps
# MAGIC - Install the Kaggle API library
# MAGIC - Move the `kaggle.json` secret for the right path
# MAGIC - Download the dataset using the API
# MAGIC - Unzip the dataset into the `../data/` folder
# MAGIC
# MAGIC Reference
# MAGIC - https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC cp ../kaggle.json /root/.kaggle/kaggle.json
# MAGIC chmod 600 /root/.kaggle/kaggle.json

# COMMAND ----------

# MAGIC %sh kaggle datasets download -d arashnic/book-recommendation-dataset -p ../data --force

# COMMAND ----------

data_path = Path("../data/")
dataset_zip = "../data/book-recommendation-dataset.zip"

with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
    zip_ref.extractall(data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Data on Feature Tables
# MAGIC
# MAGIC Once you have the dataset, the next step is to load it into your Databricks Feature Tables. This can be done using Databricks libraries.
# MAGIC
# MAGIC Reference
# MAGIC - https://docs.databricks.com/en/machine-learning/feature-store/index.html

# COMMAND ----------

books = pd.read_csv(str((data_path / "Books.csv")), low_memory=False)
books = spark.createDataFrame(books)
books.display()

# COMMAND ----------

users = pd.read_csv(str((data_path / "Users.csv")), low_memory=False)
users = spark.createDataFrame(users)
users.display()

# COMMAND ----------

ratings = pd.read_csv(str((data_path / "Ratings.csv")), low_memory=False)
ratings = spark.createDataFrame(ratings)
ratings.display()

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

fs.create_table(
    name="default.books",
    primary_keys=["ISBN"],
    df=books,
    description="Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (Book-Title, Book-Author, Year-Of-Publication, Publisher), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the Amazon web site."
)

fs.create_table(
    name="default.users",
    primary_keys=["User-ID"],
    df=users,
    description="Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Demographic data is provided (Location, Age) if available. Otherwise, these fields contain NULL-values."
)

fs.create_table(
    name="default.ratings",
    primary_keys=["User-ID", "ISBN"],
    df=ratings,
    description="Contains the book rating information. Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0."
)

# COMMAND ----------


