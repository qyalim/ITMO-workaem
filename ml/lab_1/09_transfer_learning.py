import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

SEED = 42
BATCH_SIZE = 64
MAX_TRAIN_PER_CLASS = 1500
MAX_TEST_PER_CLASS = 300
FEATURE_EPOCHS = 3
FINETUNE_EPOCHS = 2
FEATURE_LR = 1e-3
FINETUNE_LR = 1e-4

SELECTED_CLASSES = {
    3: "cat",
    5: "dog",
    7: "horse",
}

NUM_CLASSES = len(SELECTED_CLASSES)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Устройство:", device)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class ThreeClassCIFAR10(Dataset):
    def __init__(self, base_dataset, selected_classes, max_per_class=None):
        self.base_dataset = base_dataset
        old_labels = list(selected_classes.keys())
        self.label_map = {
            old_label: new_label
            for new_label, old_label in enumerate(old_labels)
        }

        indices_by_class = {label: [] for label in old_labels}

        for idx, label in enumerate(base_dataset.targets):
            if label in indices_by_class:
                indices_by_class[label].append(idx)

        rng = random.Random(SEED)
        self.indices = []

        for label in old_labels:
            indices = indices_by_class[label]
            rng.shuffle(indices)

            if max_per_class is not None:
                indices = indices[:max_per_class]

            self.indices.extend(indices)

        rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, old_label = self.base_dataset[base_idx]
        new_label = self.label_map[old_label]
        return image, new_label

train_base = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

test_base = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

train_dataset = ThreeClassCIFAR10(
    train_base,
    SELECTED_CLASSES,
    max_per_class=MAX_TRAIN_PER_CLASS,
)

test_dataset = ThreeClassCIFAR10(
    test_base,
    SELECTED_CLASSES,
    max_per_class=MAX_TEST_PER_CLASS,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

class_names = list(SELECTED_CLASSES.values())

print("Классы:", class_names)
print("Train images:", len(train_dataset))
print("Test images:", len(test_dataset))

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={epoch_loss:.4f} | "
            f"train accuracy={epoch_accuracy * 100:.2f}%"
        )

def evaluate_model(model, test_loader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(predictions)
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return y_true, y_pred, accuracy, precision, recall

def print_metrics(title, accuracy, precision, recall):
    print(f"\n--- {title} ---")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

    print("Матрица ошибок сохранена:", filename)

print("\n========================================")
print("FEATURE EXTRACTION")
print("========================================")

weights = models.ResNet18_Weights.DEFAULT
feature_model = models.resnet18(weights=weights)

for param in feature_model.parameters():
    param.requires_grad = False

num_features = feature_model.fc.in_features
feature_model.fc = nn.Linear(num_features, NUM_CLASSES)
feature_model = feature_model.to(device)

criterion = nn.CrossEntropyLoss()

feature_optimizer = torch.optim.Adam(
    feature_model.fc.parameters(),
    lr=FEATURE_LR,
)

train_model(
    feature_model,
    train_loader,
    criterion,
    feature_optimizer,
    FEATURE_EPOCHS,
)

(
    feature_y_true,
    feature_y_pred,
    feature_accuracy,
    feature_precision,
    feature_recall,
) = evaluate_model(feature_model, test_loader)

print_metrics(
    "Feature Extraction",
    feature_accuracy,
    feature_precision,
    feature_recall,
)

save_confusion_matrix(
    feature_y_true,
    feature_y_pred,
    "Feature Extraction - Confusion Matrix",
    "feature_extraction_confusion_matrix.png",
)

print("\n========================================")
print("FINE-TUNING")
print("========================================")

fine_model = copy.deepcopy(feature_model)

for param in fine_model.parameters():
    param.requires_grad = False

for param in fine_model.layer4.parameters():
    param.requires_grad = True

for param in fine_model.fc.parameters():
    param.requires_grad = True

fine_model = fine_model.to(device)

fine_optimizer = torch.optim.Adam(
    filter(
        lambda param: param.requires_grad,
        fine_model.parameters(),
    ),
    lr=FINETUNE_LR,
)

train_model(
    fine_model,
    train_loader,
    criterion,
    fine_optimizer,
    FINETUNE_EPOCHS,
)

(
    fine_y_true,
    fine_y_pred,
    fine_accuracy,
    fine_precision,
    fine_recall,
) = evaluate_model(fine_model, test_loader)

print_metrics(
    "Fine-tuning",
    fine_accuracy,
    fine_precision,
    fine_recall,
)

save_confusion_matrix(
    fine_y_true,
    fine_y_pred,
    "Fine-tuning - Confusion Matrix",
    "fine_tuning_confusion_matrix.png",
)

print("\n========================================")
print("СРАВНЕНИЕ ПОДХОДОВ")
print("========================================")

print(
    f"{'Метод':<22}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
)

print("-" * 58)

print(
    f"{'Feature Extraction':<22}"
    f"{feature_accuracy * 100:>11.2f}%"
    f"{feature_precision * 100:>11.2f}%"
    f"{feature_recall * 100:>11.2f}%"
)

print(
    f"{'Fine-tuning':<22}"
    f"{fine_accuracy * 100:>11.2f}%"
    f"{fine_precision * 100:>11.2f}%"
    f"{fine_recall * 100:>11.2f}%"
)

delta_accuracy = (fine_accuracy - feature_accuracy) * 100

print(
    "\nИзменение Accuracy после Fine-tuning:",
    f"{delta_accuracy:+.2f} п.п."
)

if delta_accuracy > 0:
    print(
        "Вывод: дообучение верхних слоев улучшило "
        "качество модели на выбранном наборе данных."
    )
elif delta_accuracy < 0:
    print(
        "Вывод: Fine-tuning не улучшил Accuracy. "
        "Возможны переобучение или необходимость "
        "изменить learning rate/число эпох."
    )
else:
    print(
        "Вывод: по Accuracy подходы дали одинаковый результат."
    )
