# -*- coding: utf-8 -*-
import os
import pickle


class GetResources:

    
    @staticmethod
    def getModel():
        model_file = 'lib/model/flavors_of_cacao.pickle'
        file = open(model_file, 'rb')
        model = pickle.load(file)
        file.close()
        return model
    @staticmethod
    def getLookupDict():
        dict_file = 'lib/data/labelDict.pickle'
        file = open(dict_file,'rb')
        lookupDict = pickle.load(file)
        file.close()
        return lookupDict