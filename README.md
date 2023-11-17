# Databricks Recommendation System

[![GitHub](https://img.shields.io/static/v1?label=Code&message=GitHub&color=blue&style=flat-square)](https://github.com/leomaurodesenv/databricks-recommendation-system)
[![MIT license](https://img.shields.io/static/v1?label=License&message=MIT&color=blue&style=flat-square)](LICENSE)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/leomaurodesenv/dvc-luigi-nlp/continuous-integration.yml?label=Build&style=flat-square)](https://github.com/leomaurodesenv/databricks-recommendation-system/actions/workflows/continuous-integration.yml)

This project study aims to implement a **recommendation system** model using Databricks Feature Store, MLflow, and Surprise modules. The goal of the project is to build an accurate recommendation system using all tracking functionalities from Databricks environment.

To achieve this goal, the project will use [Databricks Feature Store](https://docs.databricks.com/en/machine-learning/feature-store/index.html) to manage the data and features used in the model. The Feature Store provides a centralized platform for managing the data, making it easier to track changes in the data and ensure that the recommendation system is using the most up-to-date data.

Next, the project will use the [Surprise](https://surpriselib.com/) module to build the recommendation system model. Surprise is a Python library that provides a range of algorithms for building recommendation systems. The library is easy to use and provides a range of evaluation metrics for assessing the performance of the model.

Finally, the project will use [MLflow](https://mlflow.org/docs/latest/tracking.html) to track the experimentation process and compare the performance of different models. MLflow provides a centralized platform for managing the machine learning lifecycle, allowing data scientists to track the experiments, compare the performance of different models, and deploy the best model to production.

> Note: This project only runs on Databricks

---
## Code

Download or clone this repository on Databricks.
- 📔 [Run Git operations on Databricks Repos](https://docs.databricks.com/en/repos/git-operations-with-repos.html)

How to run?
1. Fill the `kaggle.json` file with your [Kaggle API](https://www.kaggle.com/docs/api) credential
2. Run the notebooks in order

---
## Also look ~

-   License [MIT](LICENSE)
-   Created by [leomaurodesenv](https://github.com/leomaurodesenv/)