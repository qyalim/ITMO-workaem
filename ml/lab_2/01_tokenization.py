
import nltk

from nltk.tokenize import (
    sent_tokenize,
    word_tokenize,
    WordPunctTokenizer
)

# Скачиваем необходимые данные NLTK.
nltk.download("punkt_tab")

# Исходный текст, который будем анализировать.
text = (
    "Hello! My name is Alice. "
    "I'm studying machine learning at university. "
    "Python is amazing, isn't it? "
    "I can't wait to learn more about NLP!"
)

print("Исходный текст:")
print(text)

# 1. Разбиваем текст на предложения.
sentences = sent_tokenize(text)

print("\n1. Токенизация предложений:")
print(sentences)

# 2. Разбиваем текст на слова.
words = word_tokenize(text)

print("\n2. Токенизация слов:")
print(words)

# 3. Разбиваем текст на слова и знаки пунктуации.
tokenizer = WordPunctTokenizer()
punct_tokens = tokenizer.tokenize(text)

print("\n3. Токенизация слов и пунктуации:")
print(punct_tokens)

# Сравниваем количество полученных токенов.
print("\nКоличество токенов:")
print("sent_tokenize:", len(sentences))
print("word_tokenize:", len(words))
print("WordPunctTokenizer:", len(punct_tokens))
