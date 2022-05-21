import numpy as np
import pandas as pd
import os
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from preprocessing import preprocessing
from preprocessing import gives_initial_features
from xgboost import XGBRegressor



st.write(""" 
# Real State Price Prediction App
This App predict the ***Price*** of a given property type in France
""" )

st.write('---')

# Load the data set w
X_init = gives_initial_features()
X, Y = preprocessing()
#Print the first rows of the Dataset
X_sample = X[:5]
st.write(X_sample)


# Sidebar
# Header of Specicify Input Parameters
st.sidebar.header('Specify Input Parameters')




def user_input_features():
    approximate_latitude = st.sidebar.slider('approximate_latitude', X.approximate_latitude.min(), X.approximate_latitude.max())
    approximate_longitude = st.sidebar.slider('approximate_longitude', X.approximate_longitude.min(), X.approximate_longitude.max())
    size = st.sidebar.slider('size', X['size'].min(), X['size'].max())
    floor = st.sidebar.slider('floor', X.floor.min(), X.floor.max())
    land_size = st.sidebar.slider('land_size', X.land_size.min(), X.land_size.max())
    energy_performance_value = st.sidebar.slider('energy_performance_value', X.energy_performance_value.min(), X.energy_performance_value.max())
    ghg_value = st.sidebar.slider('ghg_value', X.ghg_value.min(), X.ghg_value.max())
    nb_rooms = st.sidebar.slider('nb_rooms', X.nb_rooms.min(), X.nb_rooms.max())
    nb_bedrooms = st.sidebar.slider('nb_bedrooms', X.nb_bedrooms.min(), X.nb_bedrooms.max())
    nb_bathrooms = st.sidebar.slider('nb_bathrooms', X.nb_bathrooms.min(), X.nb_bathrooms.max())
    nb_parking_places = st.sidebar.slider('nb_parking_places', X.nb_parking_places.min(), X.nb_parking_places.max())
    nb_boxes = st.sidebar.slider('nb_boxes', X.nb_boxes.min(), X.nb_boxes.max())
    nb_photos = st.sidebar.slider('nb_photos', X.nb_photos.min(), X.nb_photos.max())
    has_a_balcony = st.sidebar.slider('has_a_balcony', X.has_a_balcony.min(), X.has_a_balcony.max())
    nb_terraces = st.sidebar.slider('nb_terraces', X.nb_terraces.min(), X.nb_terraces.max())
    has_a_cellar = st.sidebar.slider('has_a_cellar', X.has_a_cellar.min(), X.has_a_cellar.max())
    has_a_garage = st.sidebar.slider('has_a_garage', X.has_a_garage.min(), X.has_a_garage.max())
    has_air_conditioning = st.sidebar.slider('has_air_conditioning', X.has_air_conditioning.min(), X.has_air_conditioning.max())
    last_floor = st.sidebar.slider('last_floor', X.last_floor.min(), X.last_floor.max())
    upper_floors = st.sidebar.slider('upper_floors', X.upper_floors.min(), X.upper_floors.max())
    Paris = st.sidebar.slider('Paris', X.Paris.min(), X.Paris.max())
    Lyon = st.sidebar.slider('Lyon', X.Lyon.min(), X.Lyon.max())
    Bordeaux = st.sidebar.slider('Bordeaux', X.Bordeaux.min(), X.Bordeaux.max())
    Nice = st.sidebar.slider('Nice', X.Nice.min(), X.Nice.max(), X.Nice.mean())
    Saint_Etienne = st.sidebar.slider('Saint-Etienne', X['Saint-Etienne'].min(), X['Saint-Etienne'].max())
    Mulhouse = st.sidebar.slider('Mulhouse', X.Mulhouse.min(), X.Mulhouse.max())
    Chambre = st.sidebar.slider('Chambre', X.Chambre.min(), X.Chambre.max(), X.Chambre.mean())
    Appartement = st.sidebar.slider('Appartement', X.Appartement.min(), X.Appartement.max())
    Maison = st.sidebar.slider('Maison', X.Maison.min(), X.Maison.max())
    Terrain = st.sidebar.slider('Terrain', X.Terrain.min(), X.Terrain.max())
    Hôtel = st.sidebar.slider('Hôtel', X.Hôtel.min(), X.Hôtel.max())

    data1 = {'approximate_latitude' : approximate_latitude,
    'approximate_longitude' : approximate_longitude,
    'size' : size,
    'floor' : floor,
    'land_size': land_size,
    'energy_performance_value' : energy_performance_value,
    'ghg_value' : ghg_value,
    'nb_rooms' : nb_rooms,
    'nb_bedrooms' : nb_bedrooms,
    'nb_bathrooms' : nb_bathrooms,
    'nb_parking_places' : nb_parking_places,
    'nb_boxes' : nb_boxes,
    'nb_photos' : nb_photos,
    'has_a_balcony' : has_a_balcony,
    'nb_terraces' : nb_terraces,
    'has_a_cellar' : has_a_cellar,
    'has_a_garage' : has_a_garage,
    'has_air_conditioning' : has_air_conditioning,
    'last_floor' : last_floor,
    'upper_floors' : upper_floors,
    'Paris' : Paris,
    'Lyon' : Lyon,
    'Bordeaux' : Bordeaux,
    'Nice' : Nice,
    'Saint-Etienne' : Saint_Etienne,
    'Mulhouse' : Mulhouse,
    'Chambre' : Chambre,
    'Appartement' : Appartement,
    'Maison' : Maison,
    'Terrain' : Terrain,
    'Hôtel' : Hôtel
    }
    features = pd.DataFrame(data1, index = [0])
    return features

df = user_input_features()


# Print Specified Input Parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')



model = XGBRegressor(random_state = 100, n_estimators =  800, learning_rate =  0.1, max_depth = 9)
model = model.fit(X, Y.values.ravel())

# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of price')
st.write(prediction)
st.write('---')


