# DSND-Disaster-Response-Pipeline


# Installation

Project was written in Python 3.6. Main libraries used were:

pandas==0.23.4

numpy==1.15.4

nltk==3.4

plotly==4.5

sqlalchemy==1.2.15

flask==1.0.2

sklearn==0.22.1

# Project Motivation

This code is written as a submission for the Udacity Data Science Nanodegree data engineering project. The purpose of this repository is to showcase the implementation of a trained machine learning model for the mulit-label classification of disaster-related messages. Data is provided by Figure Eight, in collaboration with Udacity.

This project uses NLP and Latent Semantic Analysis to attempt to categorize a series of texts to disaster-related labels (eg. Fire, Storm, Aid-related, Offers, Request etc.). A model is selected, trained, and evaluated. While there is no minimum benchmark for evaluation metrics, a weighted F1-Score was used to optimize hyperparameters during the training, and label-wise precision, recall, and F-Score are printed in the ML Prep notebook. In the end, an overall accuracy of 34% was achieved. 

## Disclaimer
This repo should be used as an example of model deployment to a web app - and not as an example for optimizing machine learning models as there is much more work to be done in this respect. More details on the limitations of this model can be found in the ML Pipeline Preparation.ipynb notebook.

# Instructions

Users may download the repo, and view the web app by running '$ python \your\path\DSND\..\app\run.py' in the command line, and then visiting 'localhost:3001' in a web browser. The .PNG files attached in this repo provide an example of what the application should look like.

# Files
\

- README.md

\app

- Run.py

\templates - special note: when executing 'app = (__Flask__)', the templates folder must be specified as template_path in order for flask to use these .html files. 

- master.html

- go.html

\data

- messages.csv

- categories.csv

- distab.db

- process_data.py

\model

- trained_mlp.pkl

- train_model.py

\notebooks

- ETL Pipeline Preparation.ipynb

- ML Pipeline Preparation.ipynb



