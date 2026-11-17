
import nltk
from nltk.corpus import brown
from sklearn.feature_extraction.text import CountVectorizer

# Скачиваем корпус Brown.
nltk.download("brown", quiet=True)

# Используем те же параметры, что в задании 4.
N = 1000
M = 50

# Получаем первые 1000 слов корпуса.
words = brown.words()[:N]

# Разбиваем слова на чанки.
chunks = []

for i in range(0, len(words), M):
    chunk = words[i:i + M]
    chunks.append(" ".join(chunk))

# Создаём словарь документов.
# Ключ — номер чанка, значение — его текст.
documents = {
    index: chunk
    for index, chunk in enumerate(chunks)
}

# Создаём векторизатор Bag of Words.
vectorizer = CountVectorizer()

# Строим терм-документную матрицу.
bow_matrix = vectorizer.fit_transform(
    list(documents.values())
)

# Получаем список уникальных слов.
feature_names = vectorizer.get_feature_names_out()

# Выводим общую информацию.
print("Количество документов:", bow_matrix.shape[0])
print("Количество уникальных слов:", bow_matrix.shape[1])
print("Размер матрицы:", bow_matrix.shape)

# Преобразуем матрицу в обычный массив.
matrix = bow_matrix.toarray()

# Выводим полную матрицу.
print("\nТерм-документная матрица:")
print(matrix)

# Выводим частоты слов в первом документе.
print("\nСлова первого документа и их частоты:")

for word, count in zip(feature_names, matrix[0]):
    if count > 0:
        print(f"{word}: {count}")

# Находим слова с наибольшей общей частотой.
word_counts = matrix.sum(axis=0)

top_indices = word_counts.argsort()[::-1][:10]

print("\n10 самых частых слов:")

for index in top_indices:
    print(
        f"{feature_names[index]}: "
        f"{word_counts[index]}"
    )
