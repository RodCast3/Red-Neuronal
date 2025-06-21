from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np

from collections import defaultdict
import os

dataset_path = "Dataset_Entrenamiento"

dataset = image_dataset_from_directory(
    dataset_path,
    labels="inferred",
    label_mode="int",
    image_size=(160, 160),
    batch_size=16,
    shuffle=False
)

X_train, y_train = [], []

for images, labels in dataset:
    X_train.append(images.numpy())
    y_train.append(labels.numpy())

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

print("Imágenes cargadas:", X_train.shape)
print("Etiquetas cargadas:", y_train.shape)

# Agrupar por clase
class_to_indices = defaultdict(list)
for idx, label in enumerate(y_train):
    class_to_indices[label].append(idx)

def generate_triplets_from_dataset(X, y, num_triplets=2200):
    triplets = []
    class_labels = list(set(y))

    for _ in range(num_triplets):
        cls = np.random.choice(class_labels)
        pos_indices = class_to_indices[cls]
        if len(pos_indices) < 2:
            continue
        anchor_idx, positive_idx = np.random.choice(pos_indices, 2, replace=False)

        neg_cls = np.random.choice([c for c in class_labels if c != cls])
        neg_indices = class_to_indices[neg_cls]
        if len(neg_indices) == 0:
            continue
        negative_idx = np.random.choice(neg_indices)

        triplet = (X[anchor_idx], X[positive_idx], X[negative_idx])
        triplets.append(triplet)

    return np.stack(triplets)


triplets = generate_triplets_from_dataset(X_train, y_train, num_triplets=2200)
print("Tripletas generadas:", triplets.shape)

output_dir = "Triplets" 
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "triplets.npy")
np.save(output_path, triplets)

print(f"Tripletas guardadas en: {output_path}")
