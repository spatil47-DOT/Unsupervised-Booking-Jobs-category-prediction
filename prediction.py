"""
Author- Suraj Prakash Patil
Date- 16/05/2024
Text Preprocessing, model predictions 
"""


import pickle
import pandas as pd
import json
import warnings
warnings.simplefilter("ignore", FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score



def predict_mpg(config):
    ##loading the model from the saved file

    pkl_filename = "model_jobs_1.pkl"
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)
    print("Model imported")
    print("____________")          
                
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    print("Data Frame created")
    print("df[Job Description]")
    print(df["Job Description"])
    

    ## Creating the TFIDF vectorization model
    train = pd.read_csv('Booking_Jobs_All_220218/Booking_Jobs_All_220218.csv')


    tfidf_vect = TfidfVectorizer(sublinear_tf=True,
                            max_df=0.95,
                            min_df=round(len(train["Job Description"])*0.01),
                            stop_words="english")
    
    tfidf_vect.fit(train["Job Description"])
    print("Vectoring done!!")
    # Make model predictions
    y_pred = model.predict(tfidf_vect.transform([df["Job Description"]][0]))
    print("predictions made")
    return str(y_pred)
