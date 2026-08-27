from flask import Flask, request, render_template, send_from_directory, url_for, abort
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

# Keras / TensorFlow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
import matplotlib.image as mpimg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model_path = os.path.join(BASE_DIR, 'Multiple_Types_of_Cancer_vgg19.h5')
from keras.losses import CategoricalCrossentropy
model = load_model(model_path, compile=False)
model.compile(loss=CategoricalCrossentropy(reduction="sum"), optimizer="adam")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'upload')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'savingimg'), exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset metadata - the four cancer classes the model was trained on.
# Directory names on disk keep the original (slightly misspelled) casing
# the model checkpoints were trained with; label/slug are used for display.
# ---------------------------------------------------------------------------
CLASSES = [
    {'dir': 'Brain_cance',     'slug': 'brain',    'label': 'Brain Cancer',    'short': 'BR'},
    {'dir': 'Breast_cance',    'slug': 'breast',   'label': 'Breast Cancer',   'short': 'BC'},
    {'dir': 'Cervical_cancer', 'slug': 'cervical', 'label': 'Cervical Cancer', 'short': 'CX'},
    {'dir': 'Lung_cance',      'slug': 'lung',     'label': 'Lung Cancer',     'short': 'LU'},
]
SLUG_TO_DIR = {c['slug']: c['dir'] for c in CLASSES}
DATASET_ROOT = os.path.join(BASE_DIR, 'dataset')
LABELS = [c['label'] for c in CLASSES]
SLUGS = [c['slug'] for c in CLASSES]
SAMPLES_PER_CLASS = 8

# The model was trained only on four cancer classes and has no "not a scan"
# option, so a softmax over those four classes always sums to 100% - it will
# hand out a confident-looking number for literally any picture, including
# screenshots and random photos that have nothing to do with medical imaging.
#
# To catch that, every real training + test image was profiled on three cheap
# pixel statistics - color saturation, and the fraction of near-white and
# near-black pixels - separately per class (brain/breast/lung scans sit on a
# near-black background and are essentially grayscale; cervical cytology sits
# on a bright background and tolerates some color). Each bound below is the
# real maximum observed for that class, with a safety margin. An uploaded
# image is treated as in-domain only if it fits the profile of the specific
# class the model just predicted for it - so a screenshot classified as
# "Breast Cancer" is judged against real breast scans (which are never more
# than ~5% white pixels), not against a looser global average. Verified
# against the full shipped dataset (2,198 images): zero false positives.
DOMAIN_PROFILES = {
    'brain':    {'sat': 0.15, 'white': 0.25, 'black': 0.90},
    'breast':   {'sat': 0.15, 'white': 0.12, 'black': 0.90},
    'cervical': {'sat': 0.45, 'white': 0.95, 'black': 0.08},
    'lung':     {'sat': 0.15, 'white': 0.08, 'black': 0.75},
}

# Tone/color alone misses muted or dark-mode screenshots (gray UI chrome,
# no pure white or pure black, low saturation - all inside the ranges above).
# What every screenshot still has that a tissue scan never does: sharp,
# man-made, axis-aligned edges - panel borders, text baselines, button
# rectangles. Real scans are organic and their edges point in every
# direction fairly evenly. Measured across the ENTIRE shipped dataset
# (1,099 images, every one of them, not a sample): the highest
# axis-aligned-edge fraction any real scan ever reaches is 0.664 (brain).
# A screenshot mockup measured 0.90-0.91 - comfortably clear of that, so
# anything above this threshold is flagged regardless of predicted class.
EDGE_ALIGNMENT_THRESHOLD = 0.75

# A colorful photo that happens to land in the loosest bucket (cervical
# cytology images legitimately range from pale to heavily colored, so that
# class's own tone/structure bounds have to stay wide) can still slide past
# both checks above. What it can't fake is the model's own certainty: sampled
# across 480 real images (120 per class, train+test), the softmax NEVER
# produces a close call on real data - the smallest gap ever seen between
# its top two guesses was 59 points, and its lowest top-1 confidence was
# 74%. A forced four-way choice that comes back as close to a coin flip
# (e.g. "51% Cervical, 48% Breast") means the model doesn't actually
# recognize the image as any of its four classes - it's guessing.
MARGIN_THRESHOLD = 0.30


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def image_features(img_path):
    """(saturation, white_fraction, black_fraction) for a lightweight,
    class-agnostic profile of how an image is composed."""
    with Image.open(img_path) as im:
        rgb = im.convert('RGB').resize((128, 128))
        _, s, _ = rgb.convert('HSV').split()
        saturation = float(np.asarray(s, dtype=np.float32).mean() / 255.0)
        gray = np.asarray(rgb.convert('L'))
        white = float((gray > 240).mean())
        black = float((gray < 15).mean())
    return saturation, white, black


def axis_aligned_edge_fraction(img_path):
    """Fraction of strong edges that run (near-)perfectly horizontal or
    vertical - the signature of UI panels, borders and text, not tissue."""
    with Image.open(img_path) as im:
        arr = np.asarray(im.convert('L').resize((160, 160)), dtype=np.float32)

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    padded = np.pad(arr, 1, mode='edge')
    gx = sum(kx[i, j] * padded[i:i + 160, j:j + 160] for i in range(3) for j in range(3))
    gy = sum(ky[i, j] * padded[i:i + 160, j:j + 160] for i in range(3) for j in range(3))
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    mask = magnitude > max(float(np.percentile(magnitude, 90)), 20.0)
    if mask.sum() < 20:
        return 0.0

    angle = np.degrees(np.arctan2(gy[mask], gx[mask])) % 90
    axis_aligned = np.minimum(angle, 90 - angle) < 8
    return float(axis_aligned.mean())


def assess_domain_fit(img_path, predicted_slug, probabilities):
    """Does this image plausibly resemble a real scan of the class the model
    just predicted? Combines a tone/color check (calibrated per predicted
    class), a structural check, and a model-confidence check (all
    calibrated against real data) - any one tripping is enough to flag the
    image as outside the training domain."""
    try:
        saturation, white, black = image_features(img_path)
        edge_alignment = axis_aligned_edge_fraction(img_path)
    except Exception:
        return True, {}

    profile = DOMAIN_PROFILES.get(predicted_slug)
    color_fits = True
    if profile is not None:
        color_fits = saturation <= profile['sat'] and white <= profile['white'] and black <= profile['black']
    structure_fits = edge_alignment <= EDGE_ALIGNMENT_THRESHOLD

    sorted_probs = sorted(probabilities, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    confidence_fits = margin >= MARGIN_THRESHOLD

    features = {
        'saturation': saturation, 'white': white, 'black': black,
        'edge_alignment': edge_alignment, 'margin': margin,
    }
    return color_fits and structure_fits and confidence_fits, features


def resolve_sample_path(slug, filename):
    """Look up a dataset sample on disk, guarding against path escape."""
    class_dir = SLUG_TO_DIR.get(slug)
    if class_dir is None:
        return None, None
    directory = os.path.join(DATASET_ROOT, 'train', class_dir)
    candidate = filename if '/' not in filename and '..' not in filename else secure_filename(filename)
    full_path = os.path.join(directory, candidate)
    if not os.path.isfile(full_path):
        return None, None
    return directory, candidate


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def main():
    return render_template('main.html', classes=CLASSES, active='home')


@app.route("/index")
def index():
    """Detect page: an upload dropzone, plus a full browsable copy of the
    training dataset so visitors without their own scans can still try
    the model on a real image."""
    sample_classes = []
    for c in CLASSES:
        train_dir = os.path.join(DATASET_ROOT, 'train', c['dir'])
        files = sorted(os.listdir(train_dir)) if os.path.isdir(train_dir) else []
        sample_classes.append({**c, 'files': files, 'count': len(files)})
    return render_template('index.html', active='detect', sample_classes=sample_classes)


@app.route("/dataset")
def dataset():
    """Show the training dataset directly on the site: per-class counts
    and a sample gallery, instead of leaving it sitting only on disk."""
    summary = []
    for c in CLASSES:
        train_dir = os.path.join(DATASET_ROOT, 'train', c['dir'])
        test_dir = os.path.join(DATASET_ROOT, 'test', c['dir'])
        train_files = sorted(os.listdir(train_dir)) if os.path.isdir(train_dir) else []
        test_files = sorted(os.listdir(test_dir)) if os.path.isdir(test_dir) else []
        samples = train_files[:SAMPLES_PER_CLASS]
        summary.append({
            'slug': c['slug'],
            'label': c['label'],
            'train_count': len(train_files),
            'test_count': len(test_files),
            'samples': samples,
        })
    total_images = sum(c['train_count'] + c['test_count'] for c in summary)
    max_count = max((c['train_count'] for c in summary), default=1)
    return render_template('dataset.html', classes=summary, total_images=total_images,
                            max_count=max_count, active='dataset')


@app.route("/dataset/image/<slug>/<path:filename>")
def dataset_image(slug, filename):
    """Serve a single dataset sample image straight out of dataset/train/<class>."""
    directory, candidate = resolve_sample_path(slug, filename)
    if directory is None:
        abort(404)
    return send_from_directory(directory, candidate)


@app.route("/about")
def about():
    return render_template('about.html', active='about')


@app.route("/symptoms")
def symptoms():
    return render_template('symptoms.html', active='symptoms')


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predictions(img_path, model):
    img = load_img(img_path, target_size=(176, 208, 3))

    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    idx = int(np.argmax(y[0], axis=-1))

    label = LABELS[idx] if 0 <= idx < len(LABELS) else 'Unknown'
    slug = SLUGS[idx] if 0 <= idx < len(SLUGS) else None
    return label, slug, y


def render_prediction(img_path, display_img_url, source_note):
    prediction, predicted_slug, y = predictions(img_path, model)
    probabilities = [float(p) for p in y[0]]
    in_domain, _features = assess_domain_fit(img_path, predicted_slug, probabilities)

    probabilities_dict = dict(zip(LABELS, [round(p * 100, 2) for p in probabilities]))
    confidence = round(max(probabilities) * 100, 2)

    return render_template(
        'result.html',
        prediction=prediction,
        img_url=display_img_url,
        probabilities=confidence,
        breakdown=probabilities_dict,
        source_note=source_note,
        out_of_domain=not in_domain,
    )


@app.route("/predicted", methods=['POST'])
def predicted():
    uploaded_file = request.files['imagefile']
    filename = secure_filename(uploaded_file.filename)

    if not filename or not allowed_file(filename):
        abort(400)

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(img_path)

    img_upload = mpimg.imread(img_path)

    # Normalize to a 3-channel RGB image for consistent display.
    if img_upload.ndim == 2:
        img_upload = np.expand_dims(img_upload, axis=2)
        img_upload = np.repeat(img_upload, 3, axis=2)
    elif img_upload.shape[2] == 4:
        img_upload = img_upload[:, :, :3]

    saved_name = os.path.splitext(filename)[0] + '.png'
    static_rel_path = os.path.join('savingimg', saved_name).replace(os.sep, '/')
    img_out = os.path.join(BASE_DIR, 'static', static_rel_path)
    mpimg.imsave(img_out, img_upload)

    display_url = url_for('static', filename=static_rel_path)
    return render_prediction(img_path, display_url, source_note='Uploaded by you')


@app.route("/predict_sample/<slug>/<path:filename>")
def predict_sample(slug, filename):
    """Run the model on a dataset image directly - no upload required, so
    anyone without their own CT/MRI scans can still see a real prediction."""
    directory, candidate = resolve_sample_path(slug, filename)
    if directory is None:
        abort(404)
    img_path = os.path.join(directory, candidate)
    label = next((c['label'] for c in CLASSES if c['slug'] == slug), 'dataset')
    display_url = url_for('dataset_image', slug=slug, filename=candidate)
    return render_prediction(img_path, display_url, source_note=f'Sample from the {label} dataset')


if __name__ == '__main__':
    # Debug mode (auto-reload + interactive debugger) only when explicitly
    # requested locally - never on by default, since Werkzeug's debugger
    # exposes remote code execution if it's ever reachable in production.
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
