# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import string
import bisect
import warnings
import re

class BuildEstimator:
    
    @staticmethod
    def createBlindTestSamples():
        
        
        try:
            os.remove('api/lib/data/fitSample.csv')
            os.remove('api/lib/data/blindedTestSample.csv')
        except OSError:
            pass
        
        # read the data from csv file which is called 'flavors_of_cacao.csv' and stored in variable which is rawData
        rawData = pd.read_csv('api/lib/data/flavors_of_cacao.csv')
        #drop the colunms which are 'REF' and 'Specific Bean Origin\nor Bar Name'
        rawData = rawData.drop(columns=['REF','Specific Bean Origin\nor Bar Name'])
        #take specified colunms include 'Company','ReviewDate','CocoaPercent','CompanyLocation','Rating','BeanType', and 'BeanOrigin'
        rawData.columns = ['Company','ReviewDate','CocoaPercent','CompanyLocation','Rating','BeanType','BeanOrigin']
        #in Company colunm split each row by '(' and strip the row
        rawData['Company'] = [x.split('(')[0].strip() for x in rawData['Company']]
        #remove the '%' and convert the rest of part into float type in CocoaPercent colunm
        rawData['CocoaPercent'] = [float(x.split('%')[0].strip()) for x in rawData['CocoaPercent']]
        #remove punctuation from each row of CompanyLocation colunm
        rawData['CompanyLocation'] = [x.translate(str.maketrans('', '', string.punctuation)) for x in rawData['CompanyLocation']]
        #replace the 'Non-breaking space'(\xa0) with nan value in BeanType colunm
        rawData['BeanType'] = rawData['BeanType'].replace('\xa0', np.nan)
        #replace the nan value with 'Unknown' in BeanType colunm
        rawData['BeanType'] = rawData['BeanType'].fillna('Unknown')
        #remove all Special symbols which are ';|/|&|\.|,|\(' in BeanType colunm
        rawData['BeanType'] = [re.split(';|/|&|\.|,|\(',x)[0].strip() for x in rawData['BeanType']]
        #replace the 'Non-breaking space'(\xa0) with nan value in BeanOrigin colunm
        rawData['BeanOrigin'] = rawData['BeanOrigin'].replace('\xa0', np.nan)
        #replace the nan value with 'Unknown' in BeanOrigin colunm
        rawData['BeanOrigin'] = rawData['BeanOrigin'].fillna('Unknown')
        #remove all Special symbols which are ';|/|&|\.|,|\(' in BeanOrigin colunm
        rawData['BeanOrigin'] = [re.split(';|/|&|\.|,|\(',x)[0].strip() for x in rawData['BeanOrigin']]

        #create a list named categoricalColumns and include 'Company','CompanyLocation','BeanType','BeanOrigin'
        categoricalColumns = ['Company','CompanyLocation','BeanType','BeanOrigin']
        
        #create an empty dictionary named labelDict
        labelDict = {}
        #iterate to send each row value (stored in categoricalColumn) from first row index to last row index in categoricalColumns
        for categoricalColumn in categoricalColumns:
            #transform non-numerical labels to numerical labels.
            labelDict[categoricalColumn] = LabelEncoder()
            #fit label encoder and return encoded labels
            rawData[categoricalColumn] = labelDict[categoricalColumn].fit_transform(rawData[categoricalColumn])
            #convert each column to be corresponding lists and return the result to 'curLbl'
            curLbl = labelDict[categoricalColumn].classes_.tolist()
            #check if there are not the value of each row is 'Unknown' in curLbl
            if 'Unknown' not in curLbl:
                #insert 'Unknown' in sorted order of 'curLbl'
                bisect.insort_left(curLbl, 'Unknown')
            #correspondingly assigned 'curLbl' to labelDict
            labelDict[categoricalColumn].classes_ = curLbl

        #create a file named labelDict.pickle stored into 'api/lib/data' folder 
        leOutput = os.path.join('api/lib/data/labelDict.pickle')
        #write the 'leOutput' file by binary mode into 'file'
        file = open(leOutput,'wb')
        #dump 'file' content to the file 'labelDict'
        pickle.dump(labelDict,file)
        #closes the opened file. A closed file cannot be read or written any more. 
        file.close()

        #remove the 'Rating' column (axis=1)
        X = rawData.drop('Rating', axis=1)
        #assign 'Rating' column to 'Y'
        Y = rawData['Rating']
        #dynamically take X and Y for training and testing
        #correspondingly assigned to XFit, XBindTest, yFit, yBlindTest
        #30% of data used to test and the rest data used to train
        XFit, XBindTest, yFit, yBlindTest = train_test_split(X,Y,test_size = 0.3)

        #set index 'y' and append XFit columns
        column_head = pd.Index(['y']).append(XFit.columns)
        #create DataFrame from 'yFit' and 'XFit' and label their head column and then assigned to 'train'
        train= pd.DataFrame(np.column_stack([yFit,XFit]),columns=column_head)
        #create DataFrame from 'yBlindTest' and 'XBindTest' and label their head column and then assigned to 'blind'
        blind=pd.DataFrame(np.column_stack([yBlindTest,XBindTest]),columns=column_head)

        #write and save train data into 'fitSample.csv' without index
        train.to_csv('api/lib/data/fitSample.csv', index=False)
        #write and save blind data into 'blindedTestSample.csv' without index
        blind.to_csv('api/lib/data/blindedTestSample.csv', index=False)

    #create Python static methods
    @staticmethod
    #create a method named 'getBestPipeline' which can accept 'X' and 'y'
    def getBestPipeline(X,y):

        #prepare a range of alpha values to test
        search_params = {'alpha':[0.01,0.05,0.1,0.5,1]}
        #It is another convinience way to create a dict on the fly, Where it is to iterate through all items in search_params
        search_params = dict(('estimator__'+k, v) for k, v in search_params.items())
        #normalize features
        search_params['normalizer'] = [None,StandardScaler()]
        #featureSelector of dictionary named search_params
        #keep 90% of conponents, and run exact full SVD and select the conponents by postprocessing
        search_params['featureSelector'] = [None,PCA(n_components=0.90, svd_solver='full')]

        #create pipeline that can be cross-validated together
        pipe = Pipeline(steps=[
            ('normalizer',None),
            ('featureSelector', None),
            ('estimator', Lasso())
        ])

        #select the optimal percentage of features with grid search
        #cv: the number of folds in a (Stratified)KFold
        #verbose: Controls the verbosity: the higher, the more messages.
        #scoring: Defining scoring strategy from metric functions
        #n_jobs: Number of jobs to run in parallel,-1 means using all processors. 
        #error_score: Value to assign to the score if an error occurs in estimator fitting.
        cv = GridSearchCV(pipe,search_params,cv=10,verbose=0,scoring='neg_mean_squared_error',n_jobs=-1,error_score=0.0)
        #fit the model with parameters X and y
        
        # data_dmatrix = xgb.DMatrix(data=X, label=y)
        # cv = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10)
        
        cv.fit(X, y)
        
        
        #return the best results for modeling
        return cv
    
    #create Python static methods
    @staticmethod
    #create a method named 'createModel' without parameter
    def createModel():
        #read the csv file(the train data) named 'fitSample.csv'
        fit = pd.read_csv('api/lib/data/fitSample.csv')
        #remove the 'y' column (axis=1) and assigned the result to 'XFit'
        XFit = fit.drop(['y'],axis=1)
        #assign 'y' column to 'yFit'
        yFit = fit['y']

        #read the csv file(the blind data) named 'blindedTestSample.csv'
        blindTest = pd.read_csv('api/lib/data/blindedTestSample.csv')
        #remove the 'y' column (axis=1) and assigned the result to 'XBlindTest'
        XBlindTest = blindTest.drop(['y'],axis=1)
        #assign 'y' column to 'yBlindTest'
        yBlindTest = blindTest['y']
        
        #optimize the model by inserting the parameters 'XFit' and 'yFit', and get the best estimator
        optimizedModel = BuildEstimator.getBestPipeline(XFit,yFit).best_estimator_
        #optimizedModel = BuildEstimator.getBestPipeline(XFit,yFit)
        #predict 'XFit' and saved into 'yPredFit'
        yPredFit = optimizedModel.predict(XFit)
        #predict 'XBlindTest' and saved into 'yPredTest'
        yPredTest = optimizedModel.predict(XBlindTest)

        #calculate the mean_squared_error of 'yFit' and 'yPredFit', then stored into 'fit_score'
        fit_score = mean_squared_error(yFit,yPredFit)
        #calculate the mean_squared_error of 'yBlindTest' and 'yPredTest', then stored into 'test_score'
        test_score = mean_squared_error(yBlindTest,yPredTest)

        #print out the result
        print("fit mse = %.2f and test mse = %.2f" %(fit_score,test_score))

        #write the 'flavors_of_cacao.pickle' file by binary mode into 'file'
        file = open('api/lib/model/flavors_of_cacao.pickle','wb')
        #dump 'file' content to the file 'optimizedModel'
        pickle.dump(optimizedModel,file)
        #closes the opened file. A closed file cannot be read or written any more.
        file.close()


if __name__ == '__main__':

    
    BuildEstimator.createBlindTestSamples()
    BuildEstimator.createModel()