"""Train a new 5-way (4 cancer classes + Not a Scan) classifier head on top
of the cached frozen VGG19 features, then save a complete model matching the
original architecture shape (VGG19 backbone -> Flatten -> Dense(5))."""
import os
import numpy as np
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODEL_DIR)  # multiple_types_of_cancer/
NUM_CLASSES = 5

data = np.load(os.path.join(MODEL_DIR, '_features_cache.npz'))
X_train, y_train = data['X_train'], data['y_train']
X_val_neg, y_val_neg = data['X_val_neg'], data['y_val_neg']
X_test, y_test = data['X_test'], data['y_test']

print('X_train:', X_train.shape, 'y_train:', y_train.shape)

# Small held-out slice of the training mix itself, for early stopping
rng = np.random.default_rng(0)
idx = rng.permutation(len(X_train))
n_val = max(1, int(len(X_train) * 0.1))
val_idx, tr_idx = idx[:n_val], idx[n_val:]

head = Sequential([
    Dense(NUM_CLASSES, activation='softmax', input_shape=(X_train.shape[1],)),
])
head.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
head.fit(
    X_train[tr_idx], y_train[tr_idx],
    validation_data=(X_train[val_idx], y_train[val_idx]),
    epochs=60, batch_size=32, verbose=2, callbacks=[early_stop],
)

print('\n=== Held-out real test set (4 real classes) ===')
loss, acc = head.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {acc*100:.2f}%  (loss={loss:.4f})')
preds = np.argmax(head.predict(X_test, verbose=0), axis=1)
true = np.argmax(y_test, axis=1)
print('Predicted as "Not a Scan" (should be 0):', int((preds == 4).sum()), '/', len(preds))
for c in range(4):
    mask = true == c
    correct = (preds[mask] == c).sum()
    print(f'  class {c}: {correct}/{mask.sum()} correct')

print('\n=== Held-out NOT-scan validation set (never seen in training) ===')
loss, acc = head.evaluate(X_val_neg, y_val_neg, verbose=0)
print(f'Not-scan validation accuracy: {acc*100:.2f}%  (loss={loss:.4f})')
neg_preds = np.argmax(head.predict(X_val_neg, verbose=0), axis=1)
print('Correctly flagged as Not a Scan:', int((neg_preds == 4).sum()), '/', len(neg_preds))
import collections
print('Misclassified as:', collections.Counter(neg_preds[neg_preds != 4].tolist()))

# -----------------------------------------------------------------------
# Assemble the full model: existing frozen backbone + new head, save as .h5
# -----------------------------------------------------------------------
existing_model = load_model(os.path.join(PROJECT_ROOT, 'Multiple_Types_of_Cancer_vgg19.h5'), compile=False)
flatten_out = existing_model.get_layer('flatten_3').output
new_output = head.layers[0](flatten_out)
full_model = Model(inputs=existing_model.input, outputs=new_output)
# Saved to the project root (where app.py loads it from) and archived
# alongside the training scripts in model/ for reference.
full_model.save(os.path.join(PROJECT_ROOT, 'Multiple_Types_of_Cancer_vgg19_v2.h5'))
full_model.save(os.path.join(MODEL_DIR, 'Multiple_Types_of_Cancer_vgg19_v2.h5'))
print('\nSaved full 5-class model to Multiple_Types_of_Cancer_vgg19_v2.h5 (project root + model/)')
