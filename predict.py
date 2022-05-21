import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from preprocessing import preprocessing # Module preprocessing


def load_model(model_path):
    """
    This function load the model
    """
    # load the model
    pickled_model = pickle.load(open(model_path, 'rb'))
    return pickled_model



def funct_predict(input_features, model) :
    """
    This functions predict the price of a given property type
    input : DataFrame containing  the input features (characteristics of a property type) 
    output : dict with the id of the property type and the predicted price
    
    """
    prediction = model.predict(input_features)
    id_pred = [(id_annonce , int(pred)) for id_annonce, pred in zip(input_features.index, prediction)]
    resultat = pd.DataFrame(id_pred)
    resultat.columns = ['id_annone', 'Price property type(€)']
    resultat.set_index('id_annone', inplace = True)
    return resultat


X, Y = preprocessing()
pickled_model_gb = load_model(r"C:\Users\modfa\Documents\NEOMA\DS Remi Perrier\Project\pkl_file\model_gb.pkl")
#funct_predict(X[:5], pickled_model_rf)

from preprocessing import gives_initial_features

X_init = gives_initial_features()

if __name__ == '__main__':
    Predictions = funct_predict(X[:10], pickled_model_gb)
    Features = X_init[:10][['approximate_latitude', 'approximate_longitude', 'size', 'city', 'property_type']]
    Output_app = pd.merge(Features, Predictions, left_index=True, right_index=True)
    print(Output_app)

   






