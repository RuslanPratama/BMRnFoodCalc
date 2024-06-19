import pandas as pd

# Contoh data
food_data = pd.read_csv('calories.csv')

# Membuat DataFrame
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
    food_items = input("Masukkan nama makanan yang dipisahkan dengan koma (contoh: Apple,Carrot,Chicken): ").split(',')
    quantities = input("Masukkan jumlah gram untuk masing-masing makanan yang dipisahkan dengan koma (contoh: 150,200,250): ").split(',')
    quantities = [float(qty) for qty in quantities]
    
    total_calories = calculate_calories(food_items, quantities)
    print(f"Total kalori untuk makanan yang dimasukkan adalah: {total_calories} kalori")

if __name__ == "__main__":
    FoodCalc()
