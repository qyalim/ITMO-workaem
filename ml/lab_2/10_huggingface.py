
import os
import numpy as np
import torch

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)

# ==========================================
# 1. НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ==========================================

set_seed(42)

MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"

# Модель и промежуточные результаты
# сохраняем отдельно от Git-репозитория.
OUTPUT_DIR = "/content/distilbert_lab2_outputs"

# Количество примеров для эксперимента.
TRAIN_SIZE = 1200
VAL_SIZE = 300
TEST_SIZE = 300

# ==========================================
# 2. ЗАГРУЗКА ДАТАСЕТА
# ==========================================

print("Загрузка датасета...")

dataset = load_dataset(DATASET_NAME)

# Перемешиваем данные и выбираем подмножества.
train_data = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(TRAIN_SIZE))
)

val_data = (
    dataset["validation"]
    .shuffle(seed=42)
    .select(range(VAL_SIZE))
)

test_data = (
    dataset["test"]
    .shuffle(seed=42)
    .select(range(TEST_SIZE))
)

print("Обучающая выборка:", len(train_data))
print("Валидационная выборка:", len(val_data))
print("Тестовая выборка:", len(test_data))

# ==========================================
# 3. ТОКЕНИЗАЦИЯ
# ==========================================

print("\nЗагрузка токенизатора...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# Функция преобразования текстов в токены.
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )

# Применяем токенизатор к трём выборкам.
train_tokens = train_data.map(
    tokenize_function,
    batched=True
)

val_tokens = val_data.map(
    tokenize_function,
    batched=True
)

test_tokens = test_data.map(
    tokenize_function,
    batched=True
)

# Динамически дополняем последовательности
# до максимальной длины в текущем батче.
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# ==========================================
# 4. ЗАГРУЗКА МОДЕЛИ
# ==========================================

print("\nЗагрузка DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "Negative",
        1: "Positive"
    },
    label2id={
        "Negative": 0,
        "Positive": 1
    }
)

# ==========================================
# 5. МЕТРИКИ КАЧЕСТВА
# ==========================================

def compute_metrics(eval_pred):

    # Модель выдаёт числовые оценки классов.
    logits = eval_pred.predictions

    # Выбираем класс с максимальной оценкой.
    predictions = np.argmax(logits, axis=-1)

    # Получаем правильные ответы.
    labels = eval_pred.label_ids

    # Вычисляем Accuracy и F1-score.
    accuracy = accuracy_score(labels, predictions)

    f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "f1": f1
    }

# ==========================================
# 6. ПАРАМЕТРЫ ОБУЧЕНИЯ
# ==========================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Две эпохи обучения.
    num_train_epochs=2,

    # Размер батча.
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    # Скорость обучения.
    learning_rate=2e-5,

    # Регуляризация.
    weight_decay=0.01,

    # Проверяем качество после каждой эпохи.
    eval_strategy="epoch",

    # Сохраняем веса после обучения вручную.
    save_strategy="no",

    # Показываем статистику обучения.
    logging_strategy="epoch",

    # Используем FP16, если доступен GPU.
    fp16=torch.cuda.is_available(),

    # Отключаем интеграции отслеживания.
    report_to="none",

    # Не отправляем модель на Hugging Face.
    push_to_hub=False,

    seed=42
)

# ==========================================
# 7. СОЗДАНИЕ TRAINER
# ==========================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokens,
    eval_dataset=val_tokens,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ==========================================
# 8. FINE-TUNING
# ==========================================

print("\nНачинаем дообучение модели...")

trainer.train()

# ==========================================
# 9. ВАЛИДАЦИЯ
# ==========================================

print("\nРезультаты валидации:")

val_results = trainer.evaluate(
    eval_dataset=val_tokens
)

print("Validation Accuracy:", val_results["eval_accuracy"])
print("Validation F1:", val_results["eval_f1"])

# ==========================================
# 10. ТЕСТИРОВАНИЕ
# ==========================================

print("\nРезультаты тестирования:")

test_results = trainer.predict(test_tokens)

test_predictions = np.argmax(
    test_results.predictions,
    axis=-1
)

test_labels = test_results.label_ids

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_f1 = f1_score(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0
)

report = classification_report(
    test_labels,
    test_predictions,
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    zero_division=0
)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1: {test_f1:.4f}")

print("\nПодробный отчёт:")
print(report)

# ==========================================
# 11. ПРОВЕРКА НА СОБСТВЕННЫХ ОТЗЫВАХ
# ==========================================

custom_reviews = [
    "This movie was absolutely fantastic and inspiring.",
    "The film was boring and a complete waste of time.",
    "I really enjoyed the acting and the beautiful story.",
    "The plot was confusing and the ending was terrible.",
    "The movie looked beautiful but the story was weak."
]

print("\nПредсказания на собственных отзывах:")

# Переключаем модель в режим инференса.
model.eval()

device = next(model.parameters()).device

inference_results = []

for review in custom_reviews:

    # Токенизируем новый текст.
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    # Отключаем вычисление градиентов.
    with torch.no_grad():
        outputs = model(**inputs)

    # Преобразуем оценки в вероятности.
    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    # Выбираем наиболее вероятный класс.
    prediction = torch.argmax(
        probabilities
    ).item()

    label = model.config.id2label[prediction]

    confidence = probabilities[prediction].item()

    print("\nТекст:", review)
    print("Класс:", label)
    print(f"Вероятность: {confidence:.4f}")

    inference_results.append(
        (review, label, confidence)
    )

# ==========================================
# 12. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Сохраняем дообученную модель.
trainer.save_model(
    os.path.join(OUTPUT_DIR, "final_model")
)

# Сохраняем метрики и результаты для отчёта.
results_file = "10_results.txt"

with open(results_file, "w", encoding="utf-8") as f:

    f.write("NLP LAB 2 - TASK 10\n\n")

    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Dataset: {DATASET_NAME}\n")

    f.write(f"Train: {TRAIN_SIZE}\n")
    f.write(f"Validation: {VAL_SIZE}\n")
    f.write(f"Test: {TEST_SIZE}\n\n")

    f.write(
        f"Validation Accuracy: "
        f"{val_results['eval_accuracy']:.4f}\n"
    )

    f.write(
        f"Validation F1: "
        f"{val_results['eval_f1']:.4f}\n"
    )

    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test F1: {test_f1:.4f}\n\n")

    f.write(report)

    f.write("\nCustom reviews:\n")

    for review, label, confidence in inference_results:
        f.write(
            f"\n{review}\n"
            f"{label}: {confidence:.4f}\n"
        )

print("\nРезультаты сохранены в", results_file)
print("Модель сохранена в", OUTPUT_DIR)
print("\nЗадание 10 выполнено!")
