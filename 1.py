import pandas as pd

# 1. Загружаем данные
print("=" * 50)
print("1. Загрузка данных")
print("=" * 50)

df = pd.read_csv("train.csv")
print(f"Датасет загружен! Размер: {df.shape}")

# 2. Выводим данные на экран
print("\n" + "=" * 50)
print("2. Первые 5 строк датасета:")
print("=" * 50)
print(df.head())

# 3. Информация о датасете
print("\n" + "=" * 50)
print("3. Информация о датасете:")
print("=" * 50)
print(df.info())