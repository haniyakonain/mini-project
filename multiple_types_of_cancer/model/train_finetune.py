"""Fine-tune the 5-way classifier (4 cancer types + Not a Scan).

Replaces the previous approach - a single linear Dense head trained on
frozen ImageNet features, with no augmentation and mismatched preprocessing
- which caused the model to memorize this dataset's specific visual
signature (file format, aspect-ratio squash, source-specific look) instead
of learning real scan content. That's why it scored 100% on data drawn from
the same narrow source and collapsed to "Not a Scan" on anything else.

Fixes applied here:
  - shared, correct preprocessing (see ../preprocess.py): square-pad instead
    of squash, proper VGG19 ImageNet preprocess_input instead of raw /255.
  - real train-time augmentation (flips/rotation/brightness/contrast/crop).
  - class weights, since breast (700) dwarfs cervical (60).
  - two-phase training: warm up the new head with the backbone frozen, then
    unfreeze block5 and fine-tune end-to-end at a low learning rate, so the
    network learns actual medical-image features instead of only ever
    seeing generic ImageNet ones.

Architecture (VGG19 -> Flatten 'flatten_3' -> Dense(5, softmax)) is kept
identical to the shipped model, so app.py needs no changes beyond pointing
at the new file and using the shared preprocessing.
"""
import os
import sys
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODEL_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODEL_DIR)
from preprocess import load_and_preprocess, TARGET_HW  # noqa: E402
from augment import augment_image  # noqa: E402

DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
NEG_DIRS = [os.path.join(MODEL_DIR, 'negatives', d) for d in ('photos', 'gray', 'screens')]

CLASSES = ['Brain_cance', 'Breast_cance', 'Cervical_cancer', 'Lung_cance']
NUM_REAL = len(CLASSES)
NOT_SCAN_IDX = NUM_REAL
NUM_CLASSES = NUM_REAL + 1
SEED = 0
BATCH = 16
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


HOLDOUT_FRAC = 0.15


def list_samples():
    """Build the training pool AND a genuine held-out test set carved out of
    dataset/train/ itself.

    dataset/test/ is NOT used for evaluation: it is byte-identical to
    dataset/train/ (same filenames, same MD5 hashes, same per-class counts -
    verified directly, not assumed). Evaluating against it - as the
    original project's train_negative_class.py / train_head.py did - just
    re-scores the model on its own training images and reports that as
    "test accuracy." That produced the earlier meaningless 99.91%/100%
    numbers. The holdout carved out here is the only trustworthy signal on
    the 4 real classes.
    """
    rng = random.Random(SEED)
    samples = []
    test_paths, test_labels = [], []
    for idx, cls in enumerate(CLASSES):
        d = os.path.join(DATASET_ROOT, 'train', cls)
        files = sorted(os.listdir(d))
        rng.shuffle(files)
        n_holdout = max(1, int(len(files) * HOLDOUT_FRAC))
        holdout_files, train_files = files[:n_holdout], files[n_holdout:]
        for f in train_files:
            samples.append((os.path.join(d, f), idx))
        for f in holdout_files:
            test_paths.append(os.path.join(d, f))
            test_labels.append(idx)

    neg_paths = []
    for d in NEG_DIRS:
        for f in sorted(os.listdir(d)):
            neg_paths.append(os.path.join(d, f))
    random.shuffle(neg_paths)
    n_val_neg = max(1, int(len(neg_paths) * 0.2))
    val_neg_paths = neg_paths[:n_val_neg]
    train_neg_paths = neg_paths[n_val_neg:]
    for p in train_neg_paths:
        samples.append((p, NOT_SCAN_IDX))
    return samples, val_neg_paths, test_paths, test_labels


def make_dataset(paths, labels, batch_size, training):
    y = np.eye(NUM_CLASSES)[np.array(labels)].astype(np.float32)

    def _load(path_tensor, label_tensor):
        path = path_tensor.numpy().decode('utf-8')
        aug = augment_image if training else None
        arr = load_and_preprocess(path, augment=aug)
        return arr.astype(np.float32), label_tensor

    def _map(path, label):
        arr, label = tf.py_function(_load, [path, label], [tf.float32, tf.float32])
        arr.set_shape((TARGET_HW[0], TARGET_HW[1], 3))
        label.set_shape((NUM_CLASSES,))
        return arr, label

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    base = VGG19(weights='imagenet', include_top=False,
                 input_shape=(TARGET_HW[0], TARGET_HW[1], 3))
    x = Flatten(name='flatten_3')(base.output)
    out = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model, base


def set_trainable(base, unfreeze_from):
    """unfreeze_from=None -> whole backbone frozen. Otherwise every layer
    at/after the first layer whose name starts with unfreeze_from becomes
    trainable (e.g. 'block5' unfreezes the last conv block onward)."""
    trainable = False
    for layer in base.layers:
        if unfreeze_from is not None and layer.name.startswith(unfreeze_from):
            trainable = True
        layer.trainable = trainable


def main():
    t0 = time.time()
    samples, val_neg_paths, test_paths, test_labels = list_samples()

    n_real = sum(1 for _, l in samples if l != NOT_SCAN_IDX)
    n_neg = sum(1 for _, l in samples if l == NOT_SCAN_IDX)
    print(f'Real-class training samples: {n_real} (holding out {len(test_paths)} real scans, '
          f'never trained on, as the genuine test set)')
    print(f'Not-scan training samples:   {n_neg} (holding out {len(val_neg_paths)} for validation)')

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * 0.1))
    val_idx, tr_idx = set(idx[:n_val].tolist()), set(idx[n_val:].tolist())
    train_samples = [samples[i] for i in sorted(tr_idx)]
    val_samples = [samples[i] for i in sorted(val_idx)]

    labels_arr = np.array([l for _, l in train_samples])
    counts = np.bincount(labels_arr, minlength=NUM_CLASSES)
    class_weight = {i: len(labels_arr) / (NUM_CLASSES * max(c, 1)) for i, c in enumerate(counts)}
    print('Class counts (train split):', dict(enumerate(counts.tolist())))
    print('Class weights:', {k: round(v, 2) for k, v in class_weight.items()})

    train_ds = make_dataset([p for p, _ in train_samples], [l for _, l in train_samples], BATCH, training=True)
    val_ds = make_dataset([p for p, _ in val_samples], [l for _, l in val_samples], BATCH, training=False)

    model, base = build_model()

    print('\n=== Phase 1: training head only (backbone frozen) ===')
    set_trainable(base, unfreeze_from=None)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS,
              callbacks=[EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
              class_weight=class_weight, verbose=2)
    print(f'Phase 1 done at {time.time()-t0:.0f}s')

    print('\n=== Phase 2: fine-tuning block5 + head ===')
    set_trainable(base, unfreeze_from='block5')
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=PHASE2_EPOCHS,
              callbacks=[
                  EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
              ],
              class_weight=class_weight, verbose=2)
    print(f'Phase 2 done at {time.time()-t0:.0f}s')

    print('\n=== Held-out real test set (4 real classes) ===')
    test_ds = make_dataset(test_paths, test_labels, BATCH, training=False)
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f'Test accuracy: {acc*100:.2f}%  (loss={loss:.4f})')
    preds = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    true = np.array(test_labels)
    print('Predicted as "Not a Scan" (should be 0):', int((preds == NOT_SCAN_IDX).sum()), '/', len(preds))
    for c in range(NUM_REAL):
        mask = true == c
        correct = (preds[mask] == c).sum()
        print(f'  class {c} ({CLASSES[c]}): {correct}/{mask.sum()} correct')

    print('\n=== Held-out NOT-scan validation set (never seen in training) ===')
    val_neg_labels = [NOT_SCAN_IDX] * len(val_neg_paths)
    val_neg_ds = make_dataset(val_neg_paths, val_neg_labels, BATCH, training=False)
    loss, acc = model.evaluate(val_neg_ds, verbose=0)
    print(f'Not-scan validation accuracy: {acc*100:.2f}%  (loss={loss:.4f})')

    out_path = os.path.join(PROJECT_ROOT, 'Multiple_Types_of_Cancer_vgg19_v3.h5')
    model.save(out_path)
    model.save(os.path.join(MODEL_DIR, 'Multiple_Types_of_Cancer_vgg19_v3.h5'))
    print(f'\nSaved fine-tuned model to {out_path}')
    print(f'Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
