"""Train-time image augmentation. Applied inside preprocess.load_and_preprocess
right after square-padding, so it works on real image content rather than
raw model tensors. Kept deliberately mild - scans have real anatomical
orientation, so no extreme rotation/shear that would produce implausible
images the model would then have to unlearn.
"""
import random

from PIL import Image, ImageEnhance


def augment_image(img):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    angle = random.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

    if random.random() < 0.5:
        w, h = img.size
        crop_frac = random.uniform(0.85, 1.0)
        cw, ch = int(w * crop_frac), int(h * crop_frac)
        left = random.randint(0, max(w - cw, 0))
        top = random.randint(0, max(h - ch, 0))
        img = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)

    return img
