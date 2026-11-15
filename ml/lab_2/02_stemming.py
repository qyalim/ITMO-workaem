
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# Создаём три стеммера для сравнения.
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")

# Набор английских слов с разными окончаниями.
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

# Выводим исходный набор слов.
print("Исходные слова:")
print(", ".join(words))

# Заголовок таблицы результатов.
print("\nСравнение трёх стеммеров:")
print(f"{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")
print("-" * 60)

# Для каждого слова получаем основу тремя способами.
for word in words:
    porter_result = porter.stem(word)
    lancaster_result = lancaster.stem(word)
    snowball_result = snowball.stem(word)

    # Выводим результаты в виде таблицы.
    print(
        f"{word:<15} "
        f"{porter_result:<15} "
        f"{lancaster_result:<15} "
        f"{snowball_result:<15}"
    )
