import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

def df_to_X_y(df): 
    X = df.loc[:, list(df.columns[2:-1])].drop(['mileage'], axis=1)
    y = df['rental_price_per_day']
    for x in X.columns[-7:]:
        X[x] = (X[x].replace({False: "yes", True: "no"})).astype(str)
    return X, y # Cette fonction sépare le dataset en X et y notre valeur cible

def num_cat(X, y):
    idx = 0
    numeric_features = []
    numeric_indices = []
    categorical_features = []
    categorical_indices = []
    for i,t in X.dtypes.iteritems():
        if ('float' in str(t)) or ('int' in str(t)) :
            numeric_features.append(i)
            numeric_indices.append(idx)
        else :
            categorical_features.append(i)
            categorical_indices.append(idx)

        idx = idx + 1
    return numeric_features, numeric_indices, categorical_features, categorical_indices

def preprocessing(numeric_features, numeric_indices, categorical_features, categorical_indices):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    categorical_features = [1] 
    categorical_transformer = Pipeline(
        steps=[('encoder', OneHotEncoder(drop='first'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_indices),
            ('cat', categorical_transformer, categorical_indices)])
    return preprocessor

def split(X, y): # Fonction pour le preprocessing de X et y 
    X = X.values
    y = y.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def encoding(X_train, X_test):
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return(X_train, X_test)


# Import du dataset
pricing = pd.read_csv("data/get_around_pricing_project.csv")

# Create our X and y
X, y = df_to_X_y(pricing)
# Find columns and positions of numerical and categorical variables
numeric_features, numeric_indices, categorical_features, categorical_indices = num_cat(X, y)
# Create a preprocessor for encoding
preprocessor = preprocessing(numeric_features, numeric_indices, categorical_features, categorical_indices)
# Train test split
X_train, X_test, y_train, y_test = split(X, y)
# Using preprocessor on X_train and X_test 
X_train, X_test = encoding(X_train, X_test)


# Création de notre modèle 
model = LinearRegression()
model.fit(X_train, y_train)
train_prediction = model.predict(X_train)
score = r2_score(y_train, train_prediction)

'''        
print("Predictions: ", train_prediction)
print("r2 score: ", score)
'''

'''
X, y = df_to_X_y(pricing.sample(1))
X = X.values
y = y.tolist()
X_transformed = preprocessor.transform(X)
print(model.predict(X_transformed))
'''


#joblib.dump(model, 'linear_regression.pkl')
#joblib.dump(preprocessor, 'preprocessor.pkl')

