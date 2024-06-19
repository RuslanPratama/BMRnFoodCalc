import pandas as pd
import numpy as np

# Definisikan parameter untuk data acak
num_samples = 4000

# Gender: 'Male' atau 'female'
gender = np.random.choice(['male', 'female'], num_samples)

# Weight: acak antara 50kg hingga 100kg untuk female, dan 60kg hingga 120kg untuk Male
weight = np.where(gender == 'female', 
                  np.random.randint(40, 90, num_samples), 
                  np.random.randint(50, 100, num_samples))

# Height: acak antara 150cm hingga 180cm untuk female, dan 160cm hingga 200cm untuk Male
height = np.where(gender == 'female', 
                  np.random.randint(150, 180, num_samples), 
                  np.random.randint(150, 200, num_samples))

# Age: acak antara 18 hingga 70 tahun
age = np.random.randint(18, 25, num_samples)

# Activity_Level: kategori dari 'Sedentary', 'Lightly active', 'Moderately active', 'Very active'
activity_level = np.random.choice(['sedentary', 'light_exercise', 'moderate_exercise', 'very_active', 'extra_active'], num_samples)

# Kalori: menghitung kalori berdasarkan rumus BMR (Basal Metabolic Rate) sederhana dan tingkat aktivitas
def calculate_calories(gender, weight, height, age, activity_level):
    if gender == 'Male':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    
    activity_multipliers = {
        'sedentary': 1.2,
        'light_exercise': 1.375,
        'moderate_exercise': 1.55,
        'very_active': 1.725,
        'extra_active': 1.9
    }
    
    calories = bmr * activity_multipliers[activity_level]
    
    return int(round(calories))

calories = [calculate_calories(g, w, h, a, al) for g, w, h, a, al in zip(gender, weight, height, age, activity_level)]

# Membuat DataFrame
data = pd.DataFrame({
    'Gender': gender,
    'Weight': weight,
    'Height': height,
    'Age': age,
    'Activity_Level': activity_level,
    'Calories': calories
})

# Menyimpan dataset ke file CSV
data.to_csv('random_data_with_daily_calories.csv', index=False)

print("Dataset berhasil dibuat dan disimpan dalam file random_dataset.csv")
