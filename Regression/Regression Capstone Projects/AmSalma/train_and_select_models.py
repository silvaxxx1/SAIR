# =========================================
# Project: Medical Cost Personal Prediction
# File: train_and_select_models.py
# Description: Trains multiple regression models, tunes hyperparameters, compares them,
# and saves the best model to models/best_model.pkl
# =========================================
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

def load_data():
    url = ('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
    df = pd.read_csv(url)
    return df

def preprocess_split(df):
    # Basic preprocessing: encode categorical features, split
    df = df.copy()
    # We'll keep 'sex' and 'smoker' as binary (0/1), use get_dummies for region
    df['sex'] = df['sex'].map({'female':0, 'male':1}).astype(int)
    df['smoker'] = df['smoker'].map({'no':0, 'yes':1}).astype(int)
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    X = df.drop('charges', axis=1)
    y = df['charges']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_pipelines():
    # numeric features scaling
    numeric_features = ['age', 'bmi', 'children']
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'  # sex, smoker, region dummies will pass through
    )

    pipelines = {
        'LinearRegression': Pipeline([('pre', preprocessor),
                                      ('model', LinearRegression())]),
        'DecisionTree': Pipeline([('pre', preprocessor),
                                  ('model', DecisionTreeRegressor(random_state=42))]),
        'RandomForest': Pipeline([('pre', preprocessor),
                                 ('model', RandomForestRegressor(random_state=42))]),
        'GradientBoosting': Pipeline([('pre', preprocessor),
                                      ('model', GradientBoostingRegressor(random_state=42))]),
        'SVR': Pipeline([('pre', preprocessor),
                         ('model', SVR())])
    }
    return pipelines

def get_param_distributions():
    param_distributions = {
        'LinearRegression': {},
        'DecisionTree': {
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2,5,10]
        },
        'RandomForest': {
            'model__n_estimators': [50,100,200],
            'model__max_depth': [None,5,10,20],
            'model__min_samples_split': [2,5,10]
        },
        'GradientBoosting': {
            'model__n_estimators': [50,100,200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3,5,8]
        },
        'SVR': {
            'model__C': [0.1, 1, 10],
            'model__gamma': ['scale','auto'],
            'model__kernel': ['rbf','poly']
        }
    }
    return param_distributions

def evaluate_models(X_train, X_test, y_train, y_test, pipelines, param_dists):
    results = {}
    for name, pipe in pipelines.items():
        print(f"\nTraining and tuning: {name}")
        params = param_dists.get(name, {})
        if params:
            search = RandomizedSearchCV(pipe, param_distributions=params, n_iter=6, cv=3, scoring='r2', random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            best_params = search.best_params_
        else:
            pipe.fit(X_train, y_train)
            best = pipe
            best_params = {}
        preds = best.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results[name] = {
            'model': best,
            'rmse': rmse,
            'r2': r2,
            'best_params': best_params
        }
        print(f"{name} -> RMSE: {rmse:.2f}, R2: {r2:.3f}")
    return results

def select_and_save_best(results, output_path='models/best_model.pkl'):
    # select by highest R2, then lowest RMSE as tiebreaker
    best_name = max(results.keys(), key=lambda n: (results[n]['r2'], -results[n]['rmse']))
    best = results[best_name]['model']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(best, output_path)
    print(f"\nBest model: {best_name}")
    print(results[best_name])
    print(f"Saved best model to {output_path}")
    return best_name, results[best_name]

def main():
    print('Loading data...')
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_split(df)
    pipelines = build_pipelines()
    param_dists = get_param_distributions()
    results = evaluate_models(X_train, X_test, y_train, y_test, pipelines, param_dists)
    best_name, best_info = select_and_save_best(results)

if __name__ == '__main__':
    main()
