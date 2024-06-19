import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
import numpy as np
import xgboost as xgb

# Memuat dataset
data = pd.read_csv('diet_dataset.csv')

# Mengubah data kategorikal menjadi numerik
encoder = LabelEncoder()
data['gender'] = encoder.fit_transform(data['gender'])
data['activity_level'] = encoder.fit_transform(data['activity_level'])

# Menambah fitur baru (BMI)
data['BMI'] = data['weight'] / (data['height']/100)**2

# Menambah interaksi fitur
data['weight_height_interaction'] = data['weight'] * data['height']
data['age_BMI_interaction'] = data['age'] * data['BMI']

# Polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(data[['weight', 'height', 'age', 'BMI']])
poly_feature_names = poly.get_feature_names_out(['weight', 'height', 'age', 'BMI'])
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

# Menggabungkan fitur polinomial ke dalam dataframe asli
data = pd.concat([data, poly_df], axis=1)

# Preprocessing data
features = data.drop(columns=['calories_needed'])
target = data['calories_needed']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Hyperparameter tuning dengan Grid Search untuk XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Memprediksi pada set pengujian
y_pred = best_model.predict(X_test)

# Menghitung metrik evaluasi
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Fungsi prediksi kalori
def predict_calories(gender, weight, height, age, activity_level):
    bmi = weight / (height/100)**2
    weight_height_interaction = weight * height
    age_BMI_interaction = age * bmi

    user_data = pd.DataFrame([[gender, weight, height, age, activity_level, bmi, weight_height_interaction, age_BMI_interaction]],
                             columns=['gender', 'weight', 'height', 'age', 'activity_level', 'BMI', 'weight_height_interaction', 'age_BMI_interaction'])

    poly_user_data = poly.transform(user_data[['weight', 'height', 'age', 'BMI']])
    poly_user_df = pd.DataFrame(poly_user_data, columns=poly_feature_names)

    user_data = pd.concat([user_data, poly_user_df], axis=1)
    user_data = user_data.reindex(columns=features.columns, fill_value=0)
    scaled_user_data = scaler.transform(user_data)
    return best_model.predict(scaled_user_data)[0]

def CalCalc():
    print("Welcome to the Diet Guide Program!")
    gender = input("Enter your gender (male/female): ").strip().lower()
    weight = float(input("Enter your weight in kg: "))
    height = float(input("Enter your height in cm: "))
    age = int(input("Enter your age: "))
    activity_level = input("Enter your activity level (sedentary, light_exercise, moderate_exercise, very_active, extra_active): ").strip().lower()

    gender = 1 if gender == 'male' else 0
    activity_mapping = {'sedentary': 0, 'light_exercise': 1, 'moderate_exercise': 2, 'very_active': 3, 'extra_active': 4}
    activity_level = activity_mapping.get(activity_level, 0)

    predicted_calories = predict_calories(gender, weight, height, age, activity_level)
    print(f"\nYour predicted daily caloric needs are: {predicted_calories:.2f} calories/day")

if __name__ == "__main__":
    CalCalc()
