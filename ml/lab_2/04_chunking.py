
import nltk
from nltk.corpus import brown

# Скачиваем текстовый корпус Brown.
nltk.download("brown", quiet=True)

# Количество слов, которые берём из корпуса.
N = 1000

# Количество слов в одном чанке.
M = 50

# Загружаем первые 1000 слов.
words = brown.words()[:N]

# Разбиваем текст на чанки по 50 слов.
chunks = []

for i in range(0, len(words), M):
    chunk = words[i:i + M]

    # Объединяем слова в одну строку.
    chunk_text = " ".join(chunk)

    chunks.append(chunk_text)

# Выводим общую информацию.
print("Количество слов:", len(words))
print("Размер чанка:", M)
print("Всего чанков:", len(chunks))

# Выводим первые 20 чанков.
print("\nПолученные чанки:")

for i, chunk in enumerate(chunks[:20], start=1):
    print(f"Чанк {i}: {chunk[:50]}...")
