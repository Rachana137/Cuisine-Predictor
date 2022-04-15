# Cs5293sp22-project2

## Cuisine Predictor

### RACHANA VELLAMPALLI
### rachana@ou.edu

## Objective
The goal of this project is to create applications that take a list of ingredients from a user and attempts to predict the type of cuisine and similar meals. And Consider a chef who has a list of ingredients and would like to change the current meal without changing the ingredients. 
## PACKAGES REQUIRED
```bash
pipenv install pandas
pipenv install numpy
pipenv install scikit-learn
pipenv install nltk
import pandas as pd
import numpy as np
import json
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
```
## Steps involved 
> 1. Load json data and convert it into a dataframe
> 2. Converting "ingredients" into strings along with input ingredients.
> 3. converting ingredients into document-term matrix.
> 4.Predicting the closest cuisine and it's score.
> 5.Using cosine similarity finding top n scores and respective ID's with respect to input ingredients.
> 6. Return output in json format.

