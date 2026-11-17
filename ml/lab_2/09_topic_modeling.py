
import nltk

from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora

# Скачиваем английские стоп-слова.
nltk.download("stopwords", quiet=True)

# Создаём токенизатор.
tokenizer = RegexpTokenizer(r"\w+")

# Создаём английский стеммер.
stemmer = SnowballStemmer("english")

# Получаем набор английских стоп-слов.
stop_words = set(stopwords.words("english"))

# Путь к исходному текстовому файлу.
file_path = Path(__file__).with_name("lab02_data.txt")

# Загружаем текст из файла.
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Каждую непустую строку считаем документом.
documents = [
    line.strip()
    for line in text.splitlines()
    if line.strip()
]

print("Количество документов:", len(documents))

# Здесь будут обработанные токены.
tokens = []

# Обрабатываем каждый документ отдельно.
for document in documents:

    # Приводим текст к нижнему регистру
    # и разбиваем на отдельные слова.
    words = tokenizer.tokenize(document.lower())

    # Удаляем стоп-слова и выполняем стемминг.
    processed_words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
        and len(word) > 1
        and not word.isdigit()
    ]

    tokens.append(processed_words)

# Выводим обработанные документы.
print("\nТокены после предобработки:")

for i, token_list in enumerate(tokens, start=1):
    print(f"Документ {i}: {token_list}")

# Создаём словарь уникальных токенов.
dict_tokens = corpora.Dictionary(tokens)

print("\nРазмер словаря:", len(dict_tokens))

# Создаём терм-документную матрицу
# в разреженном формате Bag of Words.
doc_term_mat = [
    dict_tokens.doc2bow(token)
    for token in tokens
]

print("\nТерм-документная матрица:")

for i, doc in enumerate(doc_term_mat, start=1):
    print(f"Документ {i}: {doc}")

# Указываем количество тем.
num_topics = 2

# Создаём и обучаем модель LDA.
ldamodel = models.ldamodel.LdaModel(
    corpus=doc_term_mat,
    num_topics=num_topics,
    id2word=dict_tokens,
    passes=25,
    random_state=42
)

# Выводим по пять слов для каждой темы.
print("\nРЕЗУЛЬТАТЫ ТЕМАТИЧЕСКОГО МОДЕЛИРОВАНИЯ:")

for topic_id in range(num_topics):

    print(f"\nТема {topic_id + 1}:")

    topic_words = ldamodel.show_topic(
        topic_id,
        topn=5
    )

    for word, weight in topic_words:
        print(f"{word:<20} вклад: {weight:.4f}")

# Определяем распределение тем в документах.
print("\nРаспределение тем по документам:")

for i, doc in enumerate(doc_term_mat, start=1):

    topics = ldamodel.get_document_topics(
        doc,
        minimum_probability=0
    )

    print(f"\nДокумент {i}:")

    for topic_id, probability in topics:
        print(
            f"Тема {topic_id + 1}: "
            f"{probability:.4f}"
        )
