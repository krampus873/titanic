import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("=" * 50)
print("1. Загрузка данных")
print("=" * 50)

df = pd.read_csv("train.csv")
print(f"Датасет загружен! Размер: {df.shape}")

print("\n" + "=" * 50)
print("2. Первые 5 строк датасета:")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("3. Информация о датасете:")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("4. Количество пропущенных значений в каждом столбце:")
print("=" * 50)

missing_values = df.isnull().sum()
print(missing_values)

print("\nСтолбцы, в которых есть пропуски:")
missing_with_values = missing_values[missing_values > 0]
if len(missing_with_values) > 0:
    print(missing_with_values)
else:
    print("Пропусков нет!")

print("\n" + "=" * 50)
print("5. Заполнение пропущенных значений:")
print("=" * 50)

df_filled = df.copy()

age_median = df_filled['Age'].median()
df_filled['Age'] = df_filled['Age'].fillna(age_median)
print(f"Столбец 'Age': заполнено {df['Age'].isnull().sum()} пропусков медианой = {age_median:.1f}")

cabin_mode = df_filled['Cabin'].mode()[0] if len(df_filled['Cabin'].mode()) > 0 else "Unknown"
df_filled['Cabin'] = df_filled['Cabin'].fillna(cabin_mode)
print(f"Столбец 'Cabin': заполнено {df['Cabin'].isnull().sum()} пропусков модой = '{cabin_mode}'")

embarked_mode = df_filled['Embarked'].mode()[0]
df_filled['Embarked'] = df_filled['Embarked'].fillna(embarked_mode)
print(f"Столбец 'Embarked': заполнено {df['Embarked'].isnull().sum()} пропусков модой = '{embarked_mode}'")

print("\nПроверка: пропущенные значения после заполнения:")
print(df_filled.isnull().sum())
print("\n Все пропуски заполнены!")

print("\n" + "=" * 50)
print("6. Нормализация данных (MinMaxScaler):")
print("=" * 50)

numeric_columns = ['Age', 'Fare']

print(f"Нормализуем столбцы: {numeric_columns}")

scaler = MinMaxScaler()
df_filled[numeric_columns] = scaler.fit_transform(df_filled[numeric_columns])

print("После нормализации:")
print(f"Age: min={df_filled['Age'].min():.2f}, max={df_filled['Age'].max():.2f}")
print(f"Fare: min={df_filled['Fare'].min():.2f}, max={df_filled['Fare'].max():.2f}")

print("\n" + "=" * 50)
print("7. Преобразование категориальных данных (One-Hot Encoding):")
print("=" * 50)

categorical_columns = ['Sex', 'Embarked', 'Pclass']

print(f"Категориальные столбцы: {categorical_columns}")

df_final = pd.get_dummies(df_filled, columns=categorical_columns, drop_first=True)

print(f"После преобразования:")
print(f"Было столбцов: {df_filled.shape[1]}")
print(f"Стало столбцов: {df_final.shape[1]}")
print(f"\nНовые столбцы:")
print([col for col in df_final.columns if any(cat in col for cat in categorical_columns)])

print("\n" + "=" * 50)
print("8. Сохранение обработанных данных:")
print("=" * 50)

df_final.to_csv("processed_titanic.csv", index=False)
print(" Данные сохранены в файл 'processed_titanic.csv'")

print("\n" + "=" * 50)
print("ИТОГОВАЯ ИНФОРМАЦИЯ:")
print("=" * 50)
print(f"Размер исходного датасета: {df.shape}")
print(f"Размер обработанного датасета: {df_final.shape}")
print(f"Пропусков после обработки: {df_final.isnull().sum().sum()}")