import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

input_file = "income_data.txt"

data_rows = []
count_class1 = 0
count_class2 = 0
max_datapoints = 3000

with open(input_file, "r") as f:
    for line in f:
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if "?" in line:
            continue

        data = line.strip().split(", ")

        if data[-1] == "<=50K" and count_class1 < max_datapoints:
            data_rows.append(data)
            count_class1 += 1

        elif data[-1] == ">50K" and count_class2 < max_datapoints:
            data_rows.append(data)
            count_class2 += 1

X_raw = np.array(data_rows)

label_encoder = []
X_encoded = np.empty(X_raw.shape)

for i, item in enumerate(X_raw[0]):
    if item.isdigit():
        X_encoded[:, i] = X_raw[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X_raw[:, i])
        label_encoder.append(encoder)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

classifier = OneVsOneClassifier(LinearSVC(random_state=0))

f1_values = cross_val_score(classifier,X,y,scoring="f1_weighted",cv=3)
print("F1 score:", round(100 * f1_values.mean(), 2), "%")

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=5,stratify=y)


print("\n--- Эксперимент с SVM ---")

kernel_value = "linear"   # linear / rbf / poly
C_value = 1               # 1 / 10 / 100

test_classifier = SVC(
    kernel=kernel_value,
    C=C_value
)

accuracy = cross_val_score(
    test_classifier,
    X,
    y,
    scoring="accuracy",
    cv=3
)

precision = cross_val_score(
    test_classifier,
    X,
    y,
    scoring="precision_weighted",
    cv=3
)

recall = cross_val_score(
    test_classifier,
    X,
    y,
    scoring="recall_weighted",
    cv=3
)

f1 = cross_val_score(
    test_classifier,
    X,
    y,
    scoring="f1_weighted",
    cv=3
)

print("Kernel:", kernel_value)
print("C:", C_value)
print("Accuracy:", round(100 * accuracy.mean(), 2), "%")
print("Precision:", round(100 * precision.mean(), 2), "%")
print("Recall:", round(100 * recall.mean(), 2), "%")
print("F1:", round(100 * f1.mean(), 2), "%")

prediction_classifier = SVC(kernel="linear",C=1)
prediction_classifier.fit(X, y)

def predict_income(input_data):
    input_data_encoded = [-1] * len(input_data)
    count = 0

    for i, item in enumerate(input_data):
        if item.isdigit():
            input_data_encoded[i] = int(item)
        else:
            input_data_encoded[i] = int(label_encoder[count].transform([item])[0])
            count += 1

    input_data_encoded = np.array(input_data_encoded)
    predicted_class = prediction_classifier.predict([input_data_encoded])

    return label_encoder[-1].inverse_transform(predicted_class)[0]

low_indices = np.where(X_raw[:, -1] == "<=50K")[0]
high_indices = np.where(X_raw[:, -1] == ">50K")[0]

test_points = [
    X_raw[low_indices[0], :-1].tolist(),
    X_raw[high_indices[0], :-1].tolist(),
    X_raw[len(X_raw) // 2, :-1].tolist(),
]

real_labels = [
    X_raw[low_indices[0], -1],
    X_raw[high_indices[0], -1],
    X_raw[len(X_raw) // 2, -1],
]

print("\n Предсказания для трех точек данных")

for i, (point, real_label) in enumerate(
    zip(test_points, real_labels),
    start=1
):
    predicted_label = predict_income(point)

    print(f"\nТочка {i}:")
    print("Возраст:", point[0])
    print("Образование:", point[3])
    print("Профессия:", point[6])
    print("Часов работы в неделю:", point[12])
    print("Реальный класс:", real_label)
    print("Предсказанный класс:", predicted_label)
