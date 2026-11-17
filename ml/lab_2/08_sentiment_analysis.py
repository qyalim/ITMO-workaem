
import random
import nltk

from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.metrics import precision, recall, f_measure

# Скачиваем корпус отзывов.
nltk.download("movie_reviews", quiet=True)

# Извлекаем признаки из текста.
# Каждое слово становится ключом словаря.
def extract_features(words):
    return {
        word.lower(): True
        for word in words
        if word.isalpha()
    }


# Получаем идентификаторы положительных отзывов.
fileids_pos = movie_reviews.fileids("pos")

# Получаем идентификаторы отрицательных отзывов.
fileids_neg = movie_reviews.fileids("neg")

# Создаём признаки для положительных отзывов.
features_pos = [
    (
        extract_features(movie_reviews.words(fileids=[f])),
        "Positive"
    )
    for f in fileids_pos
]

# Создаём признаки для отрицательных отзывов.
features_neg = [
    (
        extract_features(movie_reviews.words(fileids=[f])),
        "Negative"
    )
    for f in fileids_neg
]

# Перемешиваем каждый класс отдельно.
random.seed(42)
random.shuffle(features_pos)
random.shuffle(features_neg)

# Делим данные в соотношении 80/20.
num_pos = int(len(features_pos) * 0.8)
num_neg = int(len(features_neg) * 0.8)

# Формируем обучающую выборку.
features_train = (
    features_pos[:num_pos]
    + features_neg[:num_neg]
)

# Формируем тестовую выборку.
features_test = (
    features_pos[num_pos:]
    + features_neg[num_neg:]
)

print("Обучающих отзывов:", len(features_train))
print("Тестовых отзывов:", len(features_test))

# Обучаем наивный байесовский классификатор.
classifier = NaiveBayesClassifier.train(
    features_train
)

# Вычисляем общую точность.
acc = nltk_accuracy(classifier, features_test)

print(f"\nAccuracy: {acc:.4f}")
print(f"Accuracy (%): {acc * 100:.2f}%")

# Для дополнительных метрик собираем
# правильные и предсказанные метки.
reference = {
    "Positive": set(),
    "Negative": set()
}

predicted = {
    "Positive": set(),
    "Negative": set()
}

for i, (features, actual_label) in enumerate(features_test):

    predicted_label = classifier.classify(features)

    reference[actual_label].add(i)
    predicted[predicted_label].add(i)

# Вычисляем Precision, Recall и F1-score.
print("\nМетрики по классам:")

for label in ["Positive", "Negative"]:

    p = precision(reference[label], predicted[label])
    r = recall(reference[label], predicted[label])
    f1 = f_measure(reference[label], predicted[label])

    print(f"\nКласс: {label}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-score:  {f1:.4f}")

# Показываем 20 наиболее информативных признаков.
print("\n20 наиболее информативных слов:")

classifier.show_most_informative_features(20)

# Собственные отзывы для проверки.
test_reviews = [
    "This movie was absolutely wonderful and exciting.",
    "The plot was boring and the acting was terrible.",
    "I loved the characters and the beautiful story.",
    "The film was disappointing and a waste of time.",
    "The movie had great music but a weak ending."
]

print("\nПредсказания на новых отзывах:")

for review in test_reviews:

    # Превращаем предложение в признаки.
    features = extract_features(review.split())

    # Получаем распределение вероятностей.
    probabilities = classifier.prob_classify(features)

    # Выбираем наиболее вероятный класс.
    prediction = probabilities.max()

    positive_prob = probabilities.prob("Positive")
    negative_prob = probabilities.prob("Negative")

    print("\nОтзыв:", review)
    print("Тональность:", prediction)
    print(f"Positive: {positive_prob:.4f}")
    print(f"Negative: {negative_prob:.4f}")
