import pytest
import project2 as p
import pandas as pd

def test_convert_string():
    data=[['romaine lettuce', 'black olives', 'grape tomatoes'],['plain flour', 'ground pepper', 'salt', 'tomatoes'],
          ['eggs', 'pepper', 'salt', 'mayonaise', 'cooking oil']]
    input_ingredients=l=['paprika','banana',"rice krispies"]
    string = p.convert_string(data,input_ingredients)
    assert type(string)==list
    assert type(string[0])==str

def test_document_matrix():
    corpus=["The fox jumps over the dog",
            "The fox is very clever and quick",
            "The dog is slow and lazy",
            "The cat is smarter than the fox and the dog"]
    matrix=p.document_matrix(corpus)
    assert type(matrix)!=str,list

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
    print(df.index.size)
    sim=p.similarity(df,matrix,1)
    assert len(sim)!=None

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
    
