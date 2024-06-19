import streamlit as st
from dailycalcalc import predict_calories
import pandas as pd

def main():
    st.title("Diet Guide Program")
    
    # Halaman pertama: Pilihan untuk menjalankan CalCalc atau FoodCalc
    page = st.sidebar.selectbox("Choose a page", ["Home", "BMR Calculator", "Calories Calculator"])

    if page == "Home":
        st.write("Welcome to Diet Guide Program!")
        st.write("Please select a page to continue.")

    elif page == "BMR Calculator":
        CalCalc()

    elif page == "Calories Calculator":
        FoodCalc()

def CalCalc():
    st.title("BMR Calculator")

    gender = st.radio("Select your gender:", ('Male', 'Female'))
    weight = st.number_input("Enter your weight in kg:", min_value=0.0, step=0.1)
    height = st.number_input("Enter your height in cm:", min_value=0.0, step=0.1)
    age = st.number_input("Enter your age:", min_value=0, step=1)
    activity_level = st.selectbox("Select your activity level:", 
                                  ['Sedentary', 'Light Exercise', 'Moderate Exercise', 'Very Active', 'Extra Active'])

    if st.button("Calculate"):
        gender_encoded = 1 if gender == 'Male' else 0
        activity_mapping = {'Sedentary': 0, 'Light Exercise': 1, 'Moderate Exercise': 2, 'Very Active': 3, 'Extra Active': 4}
        activity_level_encoded = activity_mapping[activity_level]

        predicted_calories = predict_calories(gender_encoded, weight, height, age, activity_level_encoded)
        st.success(f"Your predicted daily caloric needs are: {predicted_calories:.2f} calories/day")

food_data = pd.read_csv('calories.csv')

df = pd.DataFrame(food_data)

def calculate_calories(food_items, quantities):
    total_calories = 0
    for food, qty in zip(food_items, quantities):
        filtered_df = df[df['FoodItem'] == food]
        if not filtered_df.empty:
            cal_per_100g = filtered_df['Cals_per100grams'].values[0]
            total_calories += (cal_per_100g * qty) / 100
        else:
            print(f"Makanan '{food}' tidak ditemukan dalam dataset.")
    return total_calories

def FoodCalc():
    st.title("Food Calorie Calculator")
    FoodItem_list = df['FoodItem'].unique()

    # Tambahkan pilihan untuk melihat seluruh daftar
    if st.checkbox("Show entire food item list"):
        st.write(FoodItem_list)

    food_items = st.text_input("Enter food items seperated by comma (e.g: Apple,Carrot,Chicken):").split(',')
    quantities = st.text_input("Enter quantities in grams for each food item separated by comma (e.g., 150,200,250):").split(',')

    if st.button("Calculate Calories"):
        quantities = [float(qty) for qty in quantities]

        total_calories = calculate_calories(food_items, quantities)
        st.success(f"Total kalori untuk makanan yang dimasukkan adalah: {total_calories} kalori")


if __name__ == "__main__":
    main()
