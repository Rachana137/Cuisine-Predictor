# Cs5293sp22-project2

## Cuisine Predictor

### RACHANA VELLAMPALLI
### rachana@ou.edu

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
import argparse
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
> 4. Predicting the closest cuisine and it's score.
> 5. Using cosine similarity finding top n scores and respective ID's with respect to input ingredients.
> 6. Return output in json format.


## Project2.py
This file contains the funtions to perfom all the steps required.
1. **read_data(path)**

    This function takes the data from the given path. Here it is "yummly.json" and returns it as a dataframe using pandas.
2. **convert__string(data,input_ingredients)**

    This functions takes in the ingredients data and the given input ingredients and converts it into a string words sperated by spaces.
    **Steps**
    
    - As there are several ingredients there could be slight variations in the dataset.So, Normalizing the ingredients using PorterStemmer function. For example, Onion and Onions both convert into Onion. This will clean the data and will give more accurate data.
    
    - Next step, Before converting the list of ingredients to a list joining more than one word name ingredients by "-".
    
    The ingredients list is like this:
    ```bash
                new_ingredients
	romaine-lettuc black-ol grape-tomato garlic pe...
	plain-flour ground-pepp salt tomato ground-bla...
	egg pepper salt mayonais cooking-oil green-chi...
	water vegetable-oil wheat salt
	black-pepp shallot cornflour cayenne-pepp onio...
    ...	...	...	...	...
	light-brown-sugar granulated-sugar butter warm...
	kraft-zesty-italian-dress purple-onion broccol...
	egg citrus-fruit raisin sourdough-start flour ...
	boneless-chicken-skinless-thigh minced-garl st...
	green-chil jalapeno-chili onion ground-black-p...

    ```
3. **document_matrix(x)**

    This function performs vectorization using TfIDFvectorizer to get a document-term matrix.
    ```bash
    def document_matrix(x):
        tfidfvectorizer= TfidfVectorizer(min_df=1)
        tfidf_matrix = tfidfvectorizer.fit_transform(x)
        return tfidf_matrix
    ```
4. **probability_dataframe(value_list, class_list)**

    This function is used to create a dataframe for the predicted cuisine and it's predicted value.
    The dataframe looks like the below one.
    ```bash
                value	cuisine
        0	0.035898	brazilian
        1	0.046897	british
        2	0.059256	cajun_creole
        3	0.021414	chinese
        4	0.062652	filipino
        5	0.035989	french
        6	0.010453	greek
        7	0.076141	indian
        8	0.012711	irish
        9	0.044305	italian
        10	0.019296	jamaican
    ```
5. **predict_cuisine(df,matrix,ing)**

    It takes in the vectorizer of ingredients, dataframe and input ingredients to predict the closest cuisine and it's score of given ingredients.
    Here, For building the model the entire dataset("yummly.json") is referred as training set.So, There are several ways to approach for modeling this dataset. In this project, the algorithm used is Logistic Regression. Compared to other models randomforest, SVC and naive baye's algorithms, Logistic Regression is predicting more accurately (i.e 77%).
    
    First, Train the model and then find the Cuisine for the given set of ingredients.The predicted variable is cuisine and the dependent variable is list of ingredients which are vectorized. 
    ```bash
        model=LogisticRegression(solver='liblinear').fit(matrix, cuisine)
    ```
    To get the predicted value:
    ```bash
        predict = model.predict(vector)
    ```
    To get score of closest cuisine i.e predicted value we do:
    ```bash
        probs = model.predict_proba(vector)
        classes = model.classes_
        dframe = probability_dataframe(probs, classes)
        cuisine_closest = dframe.nlargest(1, 'value')
        closest_score = float(round(cuisine_closest.value, 2))
    ```
    The predicted probabilty of different cuisines is stored in 'probs' and the different cuisines classes in 'classes'. And, then a dataframe is created using probability_dataframe defined function. It return The cuisine and it's highest score. 
    
6. **similarity(df,matrix,n)**

    This one is for getting the "id's" and 'scores' of the given top closest n number of cuisines.To get this, cosine similarity method is used. The scores are sorted and then to find the id's corresponding to scores for loop is used to store the id and scores in a list.
    **NOTE**: I am not using KNN classifier for getting the closest possible cuisines , I am using cosine_similarity technique for getting the similar dishes.
    
    ```bash
    def similarity(df,matrix,n):
        vector = matrix[-1]
        matrix=matrix[:-1]
        scores = cosine_similarity(vector,matrix)
        lis=scores.tolist()
        score=list(np.concatenate(lis).flat)
        df['scores']=score
        score.sort(reverse=True)
        top_n = score[:n]

        dic={}
        s=[]
        for i in range(0,len(top_n),1):
            for j in range(0,df.index.size):
                if top_n[i]==df["scores"][j]:
                    dic={"id":str(df['id'][j]),"score":round(top_n[i],2)}
                    s.append(dic)
        return s
    ```
    The output is similar to this one:
    ```bash
        [{'id': '9944', 'score': 0.45},
     {'id': '8498', 'score': 0.45},
     {'id': '28111', 'score': 0.33},
     {'id': '8882', 'score': 0.33},
     {'id': '27093', 'score': 0.32},
     {'id': '40877', 'score': 0.3},
     {'id': '34845', 'score': 0.29},
     {'id': '13474', 'score': 0.29},
     {'id': '21927', 'score': 0.29},
     {'id': '39714', 'score': 0.28}]
     ```
     
7. **output(pred,score,close)**

    This function takes the predicted cuisine, score and list of closest cuisine id's and score data and returns them in json format.
    ```bash
    def output(pred,score,close):
        dictionary={}
        dictionary['predict']=pred[0]
        dictionary['score']=score
        dictionary['closest']=close
        output=json.dumps(dictionary,indent=4)
        return output
    ```
    

## Run The Program

The program is runned by the following command in the terminal.
```bash
pipenv run python project2.py --N 5 --ingredient paprika --ingredient banana --ingredient "rice krispies"
```
The output will be returned in json format.
```bash
{
    "predict": "southern_us",
    "score": 0.16,
    "closest": [
        {
            "id": "9944",
            "score": 0.45
        },
        {
            "id": "8498",
            "score": 0.45
        },
        {
            "id": "28111",
            "score": 0.33
        },
        {
            "id": "8882",
            "score": 0.33
        },
        {
            "id": "27093",
            "score": 0.32
        }
    ]
}
```

## test_all.py

This file contains the test cases to test functions in project2.py. The test functions are created for **convert__string(), document_matrix(), probability_dataframe(), predict_cuisine()** and **similarity()**.

### Packages required 

```bash
    pipenv install pytest
    import pytest
```

1. **test_convert_string()**
    
    This function tests the convert_string function whether it returns the list of ingredients in a sentence or not and the sentences are in list or not.
    ```bash
    def test_convert_string():
        data=[['romaine lettuce', 'black olives', 'grape tomatoes'],['plain flour', 'ground pepper', 'salt', 'tomatoes'],
              ['eggs', 'pepper', 'salt', 'mayonaise', 'cooking oil']]
        input_ingredients=l=['paprika','banana',"rice krispies"]
        string = p.convert_string(data,input_ingredients)
        assert type(string)==list
        assert type(string[0])==str
    ```
    
2. **test_document_matrix()**

    This test case is for document_matrix whether it is converting given set into a matrix or not.
    ```bash
    def test_document_matrix():
        corpus=["The fox jumps over the dog",
                "The fox is very clever and quick",
                "The dog is slow and lazy",
                "The cat is smarter than the fox and the dog"]
        matrix=p.document_matrix(corpus)
        assert type(matrix)!=str,list
    ```
    
3. **test_predict_cuisine()**

    This one it to test the predict_cuisine() function whether it returns the predicted cuisine and it's score.
    The test case will pass if the cuisine is of type string and score is double score.
    ```bash
    def test_predict_cuisine():
        df=pd.DataFrame()
        data=[['romaine lettuce', 'black olives', 'grape tomatoes'],['plain flour', 'ground pepper', 'salt', 'tomatoes'],
                  ['eggs', 'pepper', 'salt', 'mayonaise', 'cooking oil']]
        input_ingredients=l=['paprika','banana',"rice krispies"]
        string = p.convert_string(data,input_ingredients)
        matrix=p.document_matrix(string)
        df['cuisine']=['southern us','Asian','French']
        df['ingredients']=data
        df['id']=[9944,10250,3560]
        cuisine=df['cuisine']
        predict,score=p.predict_cuisine(df,matrix,input_ingredients)
        assert type(predict[0])==str
        assert type(score)==float
   ```
   
4. **test_similarity()**

    This one tests the similarity() function if it returns the list of values or not.
    ```bash
    def test_similarity():
        data=[['romaine lettuce', 'black olives', 'grape tomatoes'],['plain flour', 'ground pepper', 'salt', 'tomatoes'],
                  ['eggs', 'pepper', 'salt', 'mayonaise', 'cooking oil']]
        input_ingredients=l=['paprika','banana',"rice krispies"]
        string = p.convert_string(data,input_ingredients)
        matrix=p.document_matrix(string)
        df=pd.DataFrame()

        df['cuisine']=['southern us','Asian','French']
        df['ingredients']=data
        df['id']=[9944,10250,3560]

        sim=p.similarity(df,matrix,1)
        assert len(sim)!=None
        assert type(sim)==list
    ```
    
5. **test_output()**

    This test case tests the output() function if it's returning the output in json format.
    ```bash
    def test_output():
        pred=['southern us']
        score=0.81
        close=[{'id': '9944', 'score': 0.45},
               {'id': '8498', 'score': 0.45},
               {'id': '28111', 'score': 0.33},
               {'id': '8882', 'score': 0.33},
               {'id': '27093', 'score': 0.32}]
        output=p.output(pred,score,close)
        assert type(output)==str
    ```
    
To test the testcases run:
```bash
        pipenv run pytest 
        or
        pipenv run python -m pytest
```
    


## References

https://towardsdatascience.com/predict-vs-predict-proba-scikit-learn-bdc45daa5972#:~:text=The%20predict_proba()%20method,probabilities%20for%20each%20data%20point.

https://thecraftofdata.com/2019/03/making-food-recommendations-by-analyzing-cuisines-and-ingredients
