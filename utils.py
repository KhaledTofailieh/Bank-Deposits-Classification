#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd 
import numpy as np
class resultsManagement:
    def __init__(self,path,columns=None):
        try:
            self.results = pd.read_csv(path)
        except Exception as ex:
            self.results = pd.DataFrame(columns=columns)
            
    def add_to_results(self,result):
        self.results= self.results.append({c:result[c] for c in self.results.columns},ignore_index=True)
        return self.results
    def edit_result(self,index,new_result):
        for i in new_result.items():
                self.results.loc[index,i[0]] = i[1]
    def save_results(self):
        self.results.to_csv(self.path,index=False)
        
    def drop_result(self,index):
        if not isinstance(index,list):
            index = [index]
        self.results.drop(index,axis=0,inplace=True)
        
class namesParser:
    def __init__(self):
        self.name_ptn = r'\w+'
        self.params_ptn = r'\w+\((.*)\).*'
        
    def get_element_name(self,element):
        try:
            return re.match(self.name_ptn,str(element)).group()  
        except:
            return '-'
        
    def get_element_params(self,element):  
        try:
            str1=re.sub(r'\s','',str(element))
            p= re.match(self.params_ptn,str1).group(1)
            if p != '':
                return p
            return '-'
        except Exception as ex:
            return '-'    

from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.preprocessing import LabelEncoder

class unknownImputer(BaseEstimator,TransformerMixin):
    def __init__(self,unknown_value ='unknown',nan_column=None,related_column=None):
        if nan_column== None or related_column == None:
            print('Must Specify Columns!')
            raise Error()
        self.nan_column = nan_column
        self.related_column = related_column
        self.unknown_value = unknown_value
        
    def fit(self,X,y=None):
        X = pd.DataFrame(X)
        self.most_frequent = X.groupby(self.related_column)[self.nan_column].agg(pd.Series.mode)
        return self
    
    def transform(self,X,y=None):
        X = pd.DataFrame(X)
        transformed_X = X.copy()
        transformed_X[self.nan_column] = X[[self.nan_column,self.related_column]].apply(
        lambda x: self.most_frequent[x[self.related_column]] if x[self.nan_column] == self.unknown_value else x[self.nan_column] ,axis = 1)
        return transformed_X
        
class featureSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attributes):
        self.features = attributes
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X[self.features]
    
class booleanEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,true_values=['true','yes']):
        self.true_values = r'|' .join(true_values)
        
    def fit(self,X,y=None):          
        return self
    
    def transform(self,X):
        X = pd.DataFrame(X)
        transformed_X = pd.DataFrame()
        transformed_X= X.apply(lambda x: x.str.lower().str.match(self.true_values).astype(int),axis=0)
        return transformed_X
    
class numericEncoder(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.encoders = {}
        
    def fit(self,X,y=None):
        for col in X.columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self
    
    def transform(self,X):
        X = pd.DataFrame(X)
        transformed_X = pd.DataFrame() 
        transformed_X= X.apply(lambda x: self.encoders[x.name].transform(x).astype(float),axis=0)
        return transformed_X


