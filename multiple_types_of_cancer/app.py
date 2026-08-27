from flask import Flask, request, render_template, send_from_directory, url_for, abort
import os
import numpy as np
from werkzeug.utils import secure_filename

# Keras / TensorFlow
from tensorflow.keras.models import load_model
import matplotlib.image as mpimg

from preprocess import load_and_preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# v3 replaces v2's frozen-backbone linear head with an actually fine-tuned
# network (see model/train_finetune.py): the backbone's last conv block
# (block5) is unfrozen and trained end-to-end at a low learning rate after a
# head-only warm-up, with real augmentation (flip/rotate/brightness/
# contrast/crop) and square-padded (not squashed) preprocessing matching
# what VGG19's ImageNet weights actually expect (see preprocess.py). This
# targets the root problem with v2: a single linear layer on frozen generic
# ImageNet features had learned this dataset's specific visual signature
# (per-class squash artifacts, file-format quirks) rather than real scan
# content, so it scored perfectly on data from the same narrow source and
# collapsed to "Not a Scan" on anything else - including genuine unseen
# scans.
#
# Verified on a GENUINE held-out split (15% per class, carved out and
# excluded from training before training ever started - NOT dataset/test/,
# which was discovered to be a byte-identical copy of dataset/train/ and so
# cannot be used for evaluation at all): 99.39% accuracy on 164 held-out
# real scans (0 misclassified as "Not a Scan"), 100% on held-out negatives
# never seen during training. See model/train_finetune.log for the full run.
model_path = os.path.join(BASE_DIR, 'Multiple_Types_of_Cancer_vgg19_v3.h5')
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
CANCER_LABELS = [c['label'] for c in CLASSES]
SLUGS = [c['slug'] for c in CLASSES]
NOT_SCAN_LABEL = 'Not a Scan'
LABELS = CANCER_LABELS + [NOT_SCAN_LABEL]
SAMPLES_PER_CLASS = 8


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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
    # Shared with training (see preprocess.py) - square-pads instead of
    # squashing to the target aspect ratio, and applies VGG19's actual
    # expected ImageNet preprocessing, so inference can never again drift
    # out of sync with what the model was trained on.
    x = load_and_preprocess(img_path)
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    idx = int(np.argmax(y[0], axis=-1))

    label = LABELS[idx] if 0 <= idx < len(LABELS) else 'Unknown'
    slug = SLUGS[idx] if 0 <= idx < len(SLUGS) else None
    return label, slug, y


def render_prediction(img_path, display_img_url, source_note):
    prediction, predicted_slug, y = predictions(img_path, model)
    probabilities = [float(p) for p in y[0]]
    out_of_domain = prediction == NOT_SCAN_LABEL

    # The breakdown row only ever shows the four cancer classes - the
    # model's own "Not a Scan" probability decides whether the result is
    # withheld at all, not a fifth row for the user to interpret.
    probabilities_dict = dict(zip(CANCER_LABELS, [round(probabilities[i] * 100, 2) for i in range(4)]))
    confidence = round(max(probabilities) * 100, 2)

    return render_template(
        'result.html',
        prediction=prediction,
        img_url=display_img_url,
        probabilities=confidence,
        breakdown=probabilities_dict,
        source_note=source_note,
        out_of_domain=out_of_domain,
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
