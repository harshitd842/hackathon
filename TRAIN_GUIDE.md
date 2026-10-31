# How to train your own crop-disease model, from zero

This guide takes you from a bare laptop to a working `crop_disease.onnx` that the
app can load. It assumes no machine-learning background. Every command is written
out, and every output file is explained.

---

## The big picture

Training here is five stages, and only the fourth one is "training":

```
   photos on disk
        |
        |  1. collect     you photograph leaves, one folder per class
        v
   raw/<ClassName>/*.jpg
        |
        |  2. prepare     split into train/ and val/   (prepare_data.py)
        v
   data/train/<ClassName>/*.jpg  +  data/val/<ClassName>/*.jpg
        |
        |  3. features    a frozen ImageNet network turns each image
        |                into a list of 576 numbers          (forward pass only)
        v
   576-d feature vectors
        |
        |  4. train       a small classifier head learns to map
        |                those numbers to a class            (this is the training)
        v
   head.pt
        |
        |  5. export      backbone + head stitched together and written as ONNX
        v
   crop_disease.onnx  +  class_names.json   ->  drop into the app
```

Why it is built this way: the ImageNet backbone already knows how to see edges,
textures and colour patches. We do not re-teach it. We only teach a small head
the last step, "these features mean *tomato late blight*". That is why training
takes minutes on a laptop CPU, not hours on a GPU.

---

## 0. What you need

- A laptop with **Python 3.12** (not 3.13 or newer, see Troubleshooting)
- ~2 GB of free disk for the libraries
- Photos. Start with at least **one class, 300+ images**. See the photo guide.

---

## 1. Set up the environment

Run these once. Do not skip the virtual environment, it keeps this project's
libraries separate from everything else on your machine.

**Windows (PowerShell):**
```powershell
py -3.12 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime pillow numpy
```

**macOS / Linux:**
```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime pillow numpy
```

Check it worked:
```bash
python -c "import torch, torchvision, onnx, onnxruntime, PIL, numpy; print('ok')"
```

Every time you come back to the project, activate the environment first
(`.\venv\Scripts\activate` on Windows, `source venv/bin/activate` elsewhere).

---

## 2. Collect the photos

Put every photo in a folder named after its class:

```
raw/
    Tomato___Late_blight/      IMG_001.jpg  IMG_002.jpg  ..
    Tomato___healthy/          IMG_101.jpg  IMG_102.jpg  ..
    Wheat___Yellow_rust/       IMG_201.jpg  IMG_202.jpg  ..
    Wheat___healthy/           IMG_301.jpg  IMG_302.jpg  ..
```

Rules for the folder name:

- The name **is** the class name, so write it the way the app should say it.
- Use the `Crop___Condition` convention with **three underscores**.
- Keep it consistent, `Tomato___Late_blight`, never `tomato_late_blight`.
- Give every crop a `___healthy` class as well.

How many: aim for **300+ photos per class** and **50+ different leaves**, from
different plants and different fields. See the photo guide for the angles,
light and distances to shoot.

---

## 3. Prepare the data

`prepare_data.py` copies your photos into the train/val split the trainer wants.

```bash
python prepare_data.py --raw raw --out data
```

It prints a table like this:

```
class                                  total   train    val
--------------------------------------------------------------
Tomato___Late_blight                     320     256     64
Tomato___healthy                         305     244     61
--------------------------------------------------------------
total images placed: 625
classes: 2
```

What it did:

- shuffled each class and held out 20% for validation
- refused any class with fewer than 30 images (raise or lower with `--min-images`)
- left your originals in `raw/` untouched (add `--move` to move instead of copy)

Re-running is safe: a class already present in the output is skipped.

**What validation is for.** The head is trained on `train/` and then scored on
`val/`, which it never saw. That score is the only honest one. A high score on
the training images means nothing.

---

## 4. What the training actually does

Open `train_pipeline_v2.py` alongside this section.

**Stage A, features.** The backbone runs over every image once, with no
learning, and returns 576 numbers per image. Because nothing is being learned
yet, this is just a forward pass and it is fast. Training images are passed
several times through random augmentations (flips, crops, colour jitter,
rotation), so a class of 300 photos becomes 1,500 feature rows. That is how we
get variety out of a small set.

**Stage B, the head.** A small network learns to turn those 576 numbers into a
class score. It has two layers, and it is the only thing being trained.

**Stage C, the loss.** The head is scored by how wrong it is, and the weights
are nudged to reduce that. Two details matter:

- **Class weights**: a class with 300 images does not drown out a class with
  80. Each class contributes equally.
- **Label smoothing**: the head is not allowed to become 100% certain, which
  keeps it from over-committing to a wrong answer.

After every epoch the head is scored on the validation set, and the best-scoring
version is kept. That is what gets exported, not the last one.

**Stage D, export.** The backbone and the best head are joined into one model
and written as ONNX, a format that runs anywhere without PyTorch.

---

## 5. Train

```bash
python train_pipeline_v2.py --data data --out .
```

Useful flags:

| Flag | Default | What it does |
| --- | --- | --- |
| `--data` | `data` | folder holding `train/` and `val/` |
| `--out` | `.` | where the outputs are written |
| `--img` | `96` | input size, **keep 96**, the app expects it |
| `--aug` | `5` | augmented views per training image (more = slower, often better) |
| `--epochs` | `100` | training passes |

What you will see:

```
classes: 38
extracting train features (5 augmented views per image) ..
  train 26500 feature rows, val 1140  (42s)
  epoch  10  val accuracy 88.42%  (best 88.42%)
  epoch  20  val accuracy 90.11%  (best 90.11%)
  ..
best validation accuracy: 90.09% at epoch 71
weakest classes: Orange___Haunglongbing_(Citrus_greening) 74%, ..
exported crop_disease.onnx (4.36 MB)
```

**Outputs written to `--out`:**

| File | What it is |
| --- | --- |
| `crop_disease.onnx` | the model the app loads |
| `class_names.json` | class names, in the model's output order |
| `metrics.json` | accuracy, per-class accuracy, top confusions |
| `head.pt` | the head plus feature normalisation (for reuse) |

---

## 6. Read the metrics

Open `metrics.json`. Do not stop at the top-line accuracy.

- **`best_val_accuracy`**: the honest overall number. Quote this one.
- **`per_class_accuracy`**: the number that tells you where to work next. A
  class at 60% is a class you need more photos for.
- **`top_confusions`**: which classes the model mixes up, for example
  `Tomato___Early_blight -> Tomato___Late_blight`. These pairs look alike in the
  field too, so a few mistakes there are expected. A lot of them means you need
  more, and clearer, examples of the difference.
- **`images_per_class`**: a check on balance. If one class has 900 images and
  another has 60, expect the small one to score badly.

---

## 7. Test the model

Score the exported ONNX file on a few photos it has never seen:

```bash
python predict.py --image path/to/leaf.jpg
```

Run it on at least one image from every class, and on a few real field photos
with the phone in hand, not just the lab-style images. If a real photo comes
back with a low confidence, that is the expected behaviour, and it is what the
next round of photos is for.

---

## 8. Ship it into the app

Copy the two files the app needs:

```
crop_disease.onnx   ->  the app folder
class_names.json    ->  the app folder
```

Restart the app. Nothing else changes: the app reads the class names from
`class_names.json`, so it adapts to any number of classes automatically.

If you added a new disease, also add its treatment line to `farm_tools.py` so
the advisory is specific instead of the generic fallback.

---

## 9. Adding a class later

Adding a class is a data job, not a code job:

1. Put the new photos in `raw/Wheat___Yellow_rust/` (and `raw/Wheat___healthy/`).
2. Re-run `python prepare_data.py --raw raw --out data`, the existing classes
   are skipped, the new ones are split.
3. Re-run `python train_pipeline_v2.py --data data --out .`
4. Copy the new `crop_disease.onnx` and `class_names.json` into the app.

Note that the head is retrained over **all** classes each time, so a new class
can nudge the old ones. Re-check the per-class numbers after every run.

---

## 10. Improving accuracy

In the order that gives the most for the effort:

1. **More field photos.** Real photos with real backgrounds raise field accuracy
   more than any model change. This is the whole reason the photo guide exists.
2. **Balance the classes.** Roughly equal images per class.
3. **More augmentation.** Raise `--aug` to 6 or 8.
4. **Larger input.** `--img 128` or `160` helps fine detail, but then you must
   also change the input size in the app to match, and the model gets bigger.
5. **Fine-tune the backbone.** Only if the head has plateaued and you have
   thousands of images. Unfreeze the last two backbone blocks and train them at a
   low learning rate. This is slower and needs more care.

---

## 11. Troubleshooting

**`No matching distribution found for onnxruntime`**
Your Python is too new. `onnxruntime` has no wheel for 3.13+. Install Python
3.12 and create the virtual environment with `py -3.12 -m venv venv`.

**`requirements.txt not found` / `can't open file app.py`**
You are one folder too deep. The project has a nested `hackathon-main\hackathon-main`.
`cd` into the folder that actually contains `app.py`.

**`.\venv\Scripts\activate` is blocked**
Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first, in
the same PowerShell window.

**`no images found under data/train`**
The class folders are empty or the images have an extension the script does not
recognise. Check that `data/train/<ClassName>/` actually has `.jpg` files.

**`these classes have no val folder`**
A class exists in `train/` but not in `val/`. Re-run `prepare_data.py`; if a class
has only a handful of images it may have gone entirely into `train/`.

**Accuracy is high but the app is wrong on real photos**
This is the expected lab-versus-field gap, not a bug. Collect field photos and
retrain. See the photo guide.

**Training is slow**
Feature extraction is the slow part, and it scales with `--aug`. Drop to
`--aug 2` while you are still experimenting.

---

## 12. Command cheat sheet

```bash
# once
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime pillow numpy

# every time you add photos
python prepare_data.py --raw raw --out data
python train_pipeline_v2.py --data data --out .

# test
python predict.py --image sample.jpg

# ship
copy crop_disease.onnx  <app folder>
copy class_names.json   <app folder>
```

---

## File map

| File | Role |
| --- | --- |
| `prepare_data.py` | raw photos -> train/val split |
| `train_pipeline_v2.py` | features -> head -> ONNX (this guide) |
| `predict.py` | run the ONNX model on one image |
| `crop_disease.onnx` | the trained model |
| `class_names.json` | class names in output order |
| `metrics.json` | accuracy, per-class accuracy, top confusions |
| `farm_tools.py` | treatment text per class, update when you add a class |
