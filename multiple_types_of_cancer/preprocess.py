"""Shared image preprocessing for training and inference.

This exists because the original pipeline used two DIFFERENT preprocessing
paths that quietly diverged from what VGG19's ImageNet weights actually
expect, and diverged across per-class image shapes:

  - app.py's predictions() did a naive resize straight to (176, 208), which
    squashes each class's very different native aspect ratio (cervical is
    2048x1536, lung is 419x295, breast is 540x250, brain is 512x512) into a
    different-looking distortion per class - handing the model an easy,
    spurious "shape of the squash" shortcut instead of forcing it to learn
    scan content.
  - it then scaled pixels by /255 only, never applying VGG19's expected
    ImageNet mean-subtraction/BGR conversion (tf.keras.applications.vgg19.
    preprocess_input), so even the frozen backbone's "ImageNet features"
    were being fed slightly out-of-distribution inputs.

Both training and app.py now import load_and_preprocess() from here, so
train/inference preprocessing can never drift apart again.
"""
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg19 import preprocess_input

# (height, width) - matches the model's declared input shape (176, 208, 3).
TARGET_HW = (176, 208)


def _pad_to_square(img):
    """Letterbox-pad onto a black square canvas instead of stretching, so
    the resize to TARGET_HW afterwards preserves the scan's real proportions
    instead of teaching the model a per-class squash artifact."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new('RGB', (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def load_and_preprocess(path, target_hw=TARGET_HW, augment=None):
    """Load an image file and return a float32 (H, W, 3) array ready for the
    model. `augment`, if given, is a callable(PIL.Image) -> PIL.Image applied
    after square-padding but before the final resize - used only at training
    time (see model/augment.py)."""
    img = Image.open(path).convert('RGB')
    img = _pad_to_square(img)
    if augment is not None:
        img = augment(img)
    h, w = target_hw
    img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = preprocess_input(arr)  # ImageNet BGR + mean-subtraction, matches VGG19 pretraining
    return arr
