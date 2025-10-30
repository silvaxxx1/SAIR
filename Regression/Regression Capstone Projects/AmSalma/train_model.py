# =========================================
# Project: Medical Cost Personal Prediction
# File: train_model.py
# =========================================
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def load_data():
    url = ("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
           "master/insurance.csv")
    df = pd.read_csv(url)
    return df

def preprocess(df):
    df = df.copy()
    df['sex'] = LabelEncoder().fit_transform(df['sex'])
    df['smoker'] = LabelEncoder().fit_transform(df['smoker'])
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    return df

def train_and_save_model(output_path='models/best_model.pkl'):
    df = load_data()
    df = preprocess(df)
    X = df.drop('charges', axis=1)
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42)
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10]
    }
    search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                n_iter=5, scoring='r2', cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    preds = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(best_model, output_path)

    print(f"Best params: {search.best_params_}")
    print(f"RMSE: {rmse:.2f}, R2: {r2:.2f}")
    print(f"Saved model to: {output_path}")

if __name__ == '__main__':
    train_and_save_model()
