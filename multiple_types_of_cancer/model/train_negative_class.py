"""Retrain the classifier head with a genuine 5th 'Not a Scan' class, so the
model itself learns to recognize non-medical images instead of relying on
post-hoc pixel heuristics. The VGG19 convolutional base stays frozen (as in
the original model) - only the final classifier layer is retrained, which
is why this is fast even on CPU.
"""
import os
import random
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
from keras.losses import CategoricalCrossentropy

IMG_SIZE = (176, 208)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODEL_DIR)  # multiple_types_of_cancer/
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
NEG_DIRS = [
    os.path.join(MODEL_DIR, 'negatives', 'photos'),
    os.path.join(MODEL_DIR, 'negatives', 'gray'),
    os.path.join(MODEL_DIR, 'negatives', 'screens'),
]

CLASSES = [
    ('Brain_cance', 'Brain Cancer'),
    ('Breast_cance', 'Breast Cancer'),
    ('Cervical_cancer', 'Cervical Cancer'),
    ('Lung_cance', 'Lung Cancer'),
]
NUM_REAL = len(CLASSES)
NOT_SCAN_IDX = NUM_REAL  # class 4

random.seed(0)
np.random.seed(0)


def load_image_array(path):
    img = load_img(path, target_size=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = image.img_to_array(img) / 255.0
    return x


print('Loading the existing trained model and reusing its exact frozen backbone...')
existing_model = load_model(os.path.join(PROJECT_ROOT, 'Multiple_Types_of_Cancer_vgg19.h5'), compile=False)
# 'flatten_3' is the layer right before the original 4-way Dense head -
# reusing it guarantees byte-identical features to what the shipped model
# already uses, rather than risking any mismatch from re-downloading
# ImageNet weights separately.
feature_extractor = Model(inputs=existing_model.input,
                           outputs=existing_model.get_layer('flatten_3').output)
for layer in feature_extractor.layers:
    layer.trainable = False

# -----------------------------------------------------------------------
# Collect file paths + labels
# -----------------------------------------------------------------------
samples = []  # (path, label_idx)
for idx, (dirname, _) in enumerate(CLASSES):
    train_dir = os.path.join(DATASET_ROOT, 'train', dirname)
    for f in sorted(os.listdir(train_dir)):
        samples.append((os.path.join(train_dir, f), idx))

neg_paths = []
for d in NEG_DIRS:
    for f in sorted(os.listdir(d)):
        neg_paths.append(os.path.join(d, f))
random.shuffle(neg_paths)

# Hold out 20% of negatives for a genuine validation check
n_val_neg = max(1, int(len(neg_paths) * 0.2))
val_neg_paths = neg_paths[:n_val_neg]
train_neg_paths = neg_paths[n_val_neg:]

for p in train_neg_paths:
    samples.append((p, NOT_SCAN_IDX))

print(f'Real-class training samples: {sum(1 for _, l in samples if l != NOT_SCAN_IDX)}')
print(f'Not-scan training samples:   {len(train_neg_paths)} (holding out {len(val_neg_paths)} for validation)')

random.shuffle(samples)

# -----------------------------------------------------------------------
# Extract features in batches
# -----------------------------------------------------------------------
def extract_features(paths, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        arrs = np.stack([load_image_array(p) for p in batch])
        f = feature_extractor.predict(arrs, verbose=0)
        feats.append(f)
        print(f'  extracted {min(i + batch_size, len(paths))}/{len(paths)}', end='\r')
    print()
    return np.concatenate(feats, axis=0)


print('Extracting features for training set...')
train_paths = [p for p, _ in samples]
train_labels = np.array([l for _, l in samples])
X_train = extract_features(train_paths)
y_train = np.eye(NUM_REAL + 1)[train_labels]

print('Extracting features for held-out not-scan validation set...')
X_val_neg = extract_features(val_neg_paths)
y_val_neg = np.eye(NUM_REAL + 1)[np.full(len(val_neg_paths), NOT_SCAN_IDX)]

print('Extracting features for real test set (held-out real scans)...')
test_paths, test_labels = [], []
for idx, (dirname, _) in enumerate(CLASSES):
    test_dir = os.path.join(DATASET_ROOT, 'test', dirname)
    for f in sorted(os.listdir(test_dir)):
        test_paths.append(os.path.join(test_dir, f))
        test_labels.append(idx)
X_test = extract_features(test_paths)
y_test = np.eye(NUM_REAL + 1)[np.array(test_labels)]

np.savez_compressed(
    os.path.join(MODEL_DIR, '_features_cache.npz'),
    X_train=X_train, y_train=y_train,
    X_val_neg=X_val_neg, y_val_neg=y_val_neg,
    X_test=X_test, y_test=y_test,
)
print('Saved feature cache.')
