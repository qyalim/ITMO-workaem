
import nltk
from nltk.stem import WordNetLemmatizer

# Скачиваем словарь WordNet.
# Он необходим для работы лемматизатора.
nltk.download("wordnet")

# Создаём объект лемматизатора.
lemmatizer = WordNetLemmatizer()

# Тот же набор слов, что и во втором задании.
words = [
    "running",
    "runs",
    "runner",
    "studies",
    "studying",
    "happiness",
    "beautiful",
    "connection",
    "connected",
    "calves",
    "maximum",
    "generously"
]

# Выводим исходные слова.
print("Исходные слова:")
print(", ".join(words))

# Создаём заголовок таблицы.
print("\nРезультаты лемматизации:")
print(f"{'Word':<15} {'Noun (n)':<15} {'Verb (v)':<15}")
print("-" * 45)

# Лемматизируем каждое слово в двух режимах.
for word in words:

    # Считаем слово существительным.
    noun = lemmatizer.lemmatize(word, pos="n")

    # Считаем слово глаголом.
    verb = lemmatizer.lemmatize(word, pos="v")

    # Выводим результаты.
    print(f"{word:<15} {noun:<15} {verb:<15}")
