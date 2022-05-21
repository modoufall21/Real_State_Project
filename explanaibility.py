import numpy as np
import pandas as pd
import pickle
import shap
from xgboost import XGBRegressor
from preprocessing import preprocessing

X, Y = preprocessing()


# Load the model
model_gb = r"C:\Users\modfa\Documents\NEOMA\DS Remi Perrier\Project\pkl_file\model_gb.pkl"
pickled_model_gb = pickle.load(open(model_gb, 'rb'))

explainer = shap.TreeExplainer(pickled_model_gb )
shap_values = explainer.shap_values(X,approximate=True, check_additivity=False)

# Feature importance
shap.summary_plot(shap_values, X, plot_type='bar')

# importance with directionality impact of the features.
shap.summary_plot(shap_values, X)
