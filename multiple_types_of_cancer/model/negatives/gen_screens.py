"""Generate diverse synthetic screenshot/UI mockups as negative training
examples - covers light mode, dark mode, varied layouts/colors, so the
'not a scan' class learns the general shape of UI content, not one look."""
import os
import random
from PIL import Image, ImageDraw

random.seed(42)

SCREENS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'screens')
os.makedirs(SCREENS_DIR, exist_ok=True)

BG_PALETTES = [
    (255, 255, 255), (245, 245, 248), (28, 28, 32), (18, 18, 22),
    (250, 248, 240), (20, 24, 28), (255, 250, 245), (15, 15, 20),
]
ACCENT_COLORS = [
    (70, 120, 220), (220, 90, 90), (70, 180, 120), (200, 160, 60),
    (150, 90, 220), (90, 200, 200), (230, 130, 180), (255, 140, 60),
]


def rand_gray(lo, hi):
    v = random.randint(lo, hi)
    return (v, v, v)


def gen_one(idx):
    w, h = random.choice([(360, 640), (400, 700), (500, 400), (600, 400), (420, 420)])
    bg = random.choice(BG_PALETTES)
    img = Image.new('RGB', (w, h), bg)
    d = ImageDraw.Draw(img)

    dark_bg = sum(bg) / 3 < 128
    text_color = rand_gray(190, 235) if dark_bg else rand_gray(20, 70)
    panel_color = rand_gray(35, 55) if dark_bg else rand_gray(225, 245)

    # header bar
    if random.random() < 0.8:
        header_h = random.randint(40, 80)
        d.rectangle([0, 0, w, header_h], fill=random.choice(ACCENT_COLORS + [panel_color]))

    # a handful of panels/cards
    for _ in range(random.randint(1, 4)):
        x0 = random.randint(10, w // 3)
        y0 = random.randint(80, max(90, h - 150))
        x1 = min(w - 10, x0 + random.randint(80, w // 2))
        y1 = min(h - 10, y0 + random.randint(40, 160))
        d.rectangle([x0, y0, x1, y1], fill=panel_color)

    # text-like lines (rows of short dark/light rectangles)
    n_lines = random.randint(3, 10)
    for i in range(n_lines):
        y = random.randint(90, h - 30)
        x0 = random.randint(15, w // 4)
        line_w = random.randint(60, w - x0 - 20)
        d.rectangle([x0, y, x0 + line_w, y + random.randint(6, 14)], fill=text_color)

    # buttons / icons
    for _ in range(random.randint(0, 3)):
        bx = random.randint(15, w - 100)
        by = random.randint(h - 120, h - 30)
        shape = random.choice(['rect', 'circle'])
        color = random.choice(ACCENT_COLORS)
        if shape == 'rect':
            d.rectangle([bx, by, bx + random.randint(60, 140), by + random.randint(28, 44)], fill=color)
        else:
            r = random.randint(18, 34)
            d.ellipse([bx, by, bx + r, by + r], fill=color)

    # a thin divider or two
    for _ in range(random.randint(0, 2)):
        y = random.randint(90, h - 20)
        d.line([10, y, w - 10, y], fill=rand_gray(80, 120), width=1)

    img.save(os.path.join(SCREENS_DIR, f'screen_{idx}.png'))


for i in range(120):
    gen_one(i)

print('generated 120 synthetic screenshots')
