
import random
import nltk

from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

# Скачиваем набор мужских и женских имён.
nltk.download("names", quiet=True)

# Функция извлекает последние N букв имени.
def extract_features(name, n):
    name = name.lower()

    return {
        "suffix": name[-n:]
    }


# Загружаем мужские имена.
male_list = [
    (name, "male")
    for name in names.words("male.txt")
]

# Загружаем женские имена.
female_list = [
    (name, "female")
    for name in names.words("female.txt")
]

# Объединяем данные.
data = male_list + female_list

# Устанавливаем seed для воспроизводимости.
random.seed(42)

# Перемешиваем имена.
random.shuffle(data)

# 80% данных используем для обучения,
# остальные 20% — для тестирования.
split_index = int(len(data) * 0.8)

train_data = data[:split_index]
test_data = data[split_index:]

# Имена для самостоятельной проверки.
test_names = [
    "Alexander",
    "Sophia",
    "Michael",
    "Olivia",
    "Daniel",
    "Amelia"
]

print("Всего имён:", len(data))
print("Обучающая выборка:", len(train_data))
print("Тестовая выборка:", len(test_data))

# Здесь будем хранить результаты точности.
results = {}

# Проверяем количество последних букв от 1 до 5.
for n in range(1, 6):

    # Извлекаем признаки из обучающих имён.
    train_features = [
        (extract_features(name, n), gender)
        for name, gender in train_data
    ]

    # Извлекаем признаки из тестовых имён.
    test_features = [
        (extract_features(name, n), gender)
        for name, gender in test_data
    ]

    # Обучаем наивный байесовский классификатор.
    classifier = NaiveBayesClassifier.train(
        train_features
    )

    # Вычисляем Accuracy.
    acc = nltk_accuracy(classifier, test_features)

    results[n] = acc

    print(f"\nКоличество букв N = {n}")
    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    # Проверяем модель на собственных именах.
    for name in test_names:

        features = extract_features(name, n)

        prediction = classifier.classify(features)

        print(f"{name:<12} -> {prediction}")

# Находим N с максимальной точностью.
best_n = max(results, key=results.get)

print("\nИТОГОВОЕ СРАВНЕНИЕ:")

for n, acc in results.items():
    print(f"N = {n}: {acc * 100:.2f}%")

print("\nОптимальное количество букв:", best_n)
print(f"Лучшая точность: {results[best_n] * 100:.2f}%")
