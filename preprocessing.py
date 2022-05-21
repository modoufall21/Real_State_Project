import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PreprossData :
    """
    This class load and process the data
    """
    def __init__(self, file_path_var, file_path_prix):
        """
        Instanciate the class
        """
        self.X = pd.read_csv(file_path_var, sep = ';', index_col = 'id_annonce')
        self.Y = pd.read_csv(file_path_prix, sep = ',', index_col = 'id_annonce')

    def give_features(self):
        """
        Input variables
        """
        input_data = self.X
        #input_data.set_index('id_annonce', inplace = True)
        return input_data 

    def give_target(self):
        """
        target variable
        """
        target = self.Y
        #target.set_index('id_annonce', inplace = True)
        return target

    def separate_cat_num_features(self):
        """
        Separate numerical and categorical columns
        """
        cat_features = []
        num_features = []
        X = self.give_features()
        for i in X.columns:
            if (X[str(i)].dtypes == object):
                cat_features.append(i)
            else :
                num_features.append(i)
        return {'cat_features' : cat_features, 'num_features' : num_features}


# Saint Etienne, Mulhouse, Brest, Limoge, Leman
    def features_engineering(self):
        """
        This function create the following features:
        New features :
            City :  Paris, Lyon, Bordeaux, Nice, Saint-Etienne, Mulhouse
            Property type : Chambre, Appartement, Maison, Terrain, Hôtel, Maison
        drop the features : 'property_type', 'city', 'energy_performance_category', 'ghg_category', 'exposition'

        """
        X = self.give_features()
        # Localization
        # City with hight price
        X['Paris'] = [1 if 'paris' in i else 0 for i in X['city']]
        X['Lyon'] = [1 if 'lyon' in i else 0 for i in X['city']]
        X['Bordeaux'] = [1 if 'bordeaux' in i else 0 for i in X['city']]
        X['Nice'] = [1 if 'nice' in i else 0 for i in X['city']]
        # City with low price
        X['Saint-Etienne'] = [1 if 'saint-etienne' in i else 0 for i in X['city']]
        X['Mulhouse'] = [1 if 'mulhouse' in i else 0 for i in X['city']]

        # Property type
        # 'chambre', 'appartement', 'maison','château',  'terrain', 'terrain à bâtir', 'hôtel particulier','hôtel'
        X['Chambre'] = [1 if 'chambre' in i else 0 for i in X['property_type']]
        X['Appartement'] = [1 if 'appartement' in i else 0 for i in X['property_type']]
        X['Maison'] = [1 if 'maison' in i else 0 for i in X['property_type']]
        X['Terrain'] = [1 if 'terrain' in i else 0 for i in X['property_type']]
        X['Hôtel'] = [1 if 'hôtel' in i else 0 for i in X['property_type']]
        X['Maison'] = [1 if 'maison' in i else 0 for i in X['property_type']]

        cat_features = self.separate_cat_num_features()['cat_features']
        X.drop(cat_features, axis = 1, inplace = True)
        X.drop('postal_code', axis = 1, inplace = True)
        return X
 

    def replace_missing_values(self):
        """
        This function replace missing values of the columns x by its median.
        Columns with missing values :
                size, floor, land_size, energy_performance_value, ghg_value, nb_rooms, nb_bedrooms, nb_bathrooms  
        
        """
        X = self.features_engineering()
        X.fillna(X.median(), inplace = True)
        return X




path_features = r"C:\Users\modfa\Documents\NEOMA\DS Remi Perrier\Project\data\variables.csv"
path_target = r"C:\Users\modfa\Documents\NEOMA\DS Remi Perrier\Project\data\prix.csv" 

def preprocessing(path_features = path_features , path_target = path_target ):
    """
    This function give us the final result of our module proprocessing
    """
    pc = PreprossData(path_features, path_target)
    Y = pc.give_target()
    X = pc.replace_missing_values()
    return X, Y

X, Y = preprocessing()

def gives_initial_features():
    pc = PreprossData(path_features, path_target)
    X_init = pc.give_features()
    return X_init

X_init = gives_initial_features()



if __name__ == '__main__':
    print(X.head(5))
    print(f'Input features :\nNumbers of rows : {X.shape[0]}\nNumber of Columns : {X.shape[1]}')
    print(f'Target :\nNumber of rows : {Y.shape[0]}\nNumber of Columns : {Y.shape[1]}')
    print(X['Paris'].value_counts())
    print(Y.head(5))

    print(X.isna().sum())

    


