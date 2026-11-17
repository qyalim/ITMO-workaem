
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer
)

# Выбираем пять категорий из датасета 20 Newsgroups.
categories = [
    "comp.graphics",
    "rec.sport.hockey",
    "sci.space",
    "sci.med",
    "talk.politics.mideast"
]

# Указываем понятные названия категорий.
category_names = {
    "comp.graphics": "Компьютерная графика",
    "rec.sport.hockey": "Хоккей",
    "sci.space": "Космос",
    "sci.med": "Медицина",
    "talk.politics.mideast": "Политика Ближнего Востока"
}

# Загружаем обучающие документы.
# remove убирает заголовки, подписи и цитаты.
training_data = fetch_20newsgroups(
    subset="train",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes")
)

# Преобразуем тексты в матрицу частот слов.
vectorizer = CountVectorizer()

train_counts = vectorizer.fit_transform(
    training_data.data
)

# Создаём преобразователь TF-IDF.
tfidf_transformer = TfidfTransformer()

# Рассчитываем TF-IDF для обучающих текстов.
train_tfidf = tfidf_transformer.fit_transform(
    train_counts
)

# Создаём и обучаем классификатор.
classifier = MultinomialNB()

classifier.fit(
    train_tfidf,
    training_data.target
)

# Создаём собственные предложения для проверки.
test_sentences = [
    "The graphics card renders detailed 3D images.",
    "The hockey team scored a goal in the final match.",
    "Astronauts are preparing for a mission to Mars.",
    "Doctors discovered a new treatment for the disease.",
    "The government discussed the Middle East peace agreement."
]

# Преобразуем новые предложения в числовые признаки.
# Используем УЖЕ обученный векторизатор.
test_counts = vectorizer.transform(test_sentences)

# Применяем обученный преобразователь TF-IDF.
test_tfidf = tfidf_transformer.transform(
    test_counts
)

# Предсказываем категории предложений.
predictions = classifier.predict(test_tfidf)

# Выводим категории обучения.
print("Категории классификации:")

for category in training_data.target_names:
    print("-", category_names[category])

print("\nКоличество обучающих документов:")
print(len(training_data.data))

# Выводим результаты классификации.
print("\nРезультаты предсказаний:\n")

for sentence, prediction in zip(
    test_sentences, predictions
):
    category = training_data.target_names[prediction]

    print("Текст:", sentence)
    print("Категория:", category_names[category])
    print("-" * 60)
