# Setup Guide — run this project on your own computer, then push to GitHub

This guide is for beginners. Follow it top to bottom. Commands are for **Windows**;
Mac/Linux differences are noted.

---

## Part A — Get it running on your computer (VS Code)

### 1. Install Python
- Go to https://www.python.org/downloads/ and install Python 3.11 or newer.
- **Windows:** on the first install screen, tick the box **"Add Python to PATH"**.
- Check it worked. Open a terminal (Windows: search "PowerShell") and type:
  ```
  python --version
  ```
  You should see something like `Python 3.12.x`.

### 2. Install VS Code
- Download from https://code.visualstudio.com/
- Open VS Code, go to **Extensions** (left sidebar), search **Python** (by Microsoft) and click **Install**.

### 3. Put the project files in one folder
Create a folder, e.g. `crop-disease-detector`, and put these files inside it:
```
crop_disease.onnx
class_names.json
predict.py
app.py
train_pipeline.py
model_metrics.json
requirements.txt
README.md
demo_predictions.png
.gitignore
```

### 4. Open the folder in VS Code
- VS Code → **File → Open Folder** → pick your folder.
- Open the terminal inside VS Code: **Terminal → New Terminal** (or press `` Ctrl+` ``).

### 5. Create a virtual environment (recommended)
This keeps your project's packages separate from the rest of your system.
```
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You will see `(venv)` at the start of your terminal line.

### 6. Install the packages
```
pip install -r requirements.txt
```

### 7. Run it
```
python predict.py path\to\a\leaf-photo.jpg
```
(Use any photo of a single leaf. You can download a sample leaf image from the PlantVillage
dataset or just photograph one.) It prints the top-3 predicted diseases. To open the web demo:
```
python app.py
```
A local link (like `http://127.0.0.1:7860`) appears — open it in your browser and upload a leaf photo.

> If you want to **re-train**, you also need `torch`, `torchvision` and `onnx`
> (uncomment them in `requirements.txt`), plus the dataset in a `data/` folder.

---

## Part B — Put it on GitHub

GitHub does **not run** your code — it stores it and shows it to others (great for a
hackathon portfolio). You run the code on your own machine.

### 1. Install Git
- Download from https://git-scm.com/downloads and install (default options are fine).
- Check: `git --version`

### 2. Create an empty repository on GitHub
- Log in at https://github.com → click **+** (top right) → **New repository**.
- Name it `crop-disease-detector`. **Do not** tick "Add a README" (you already have one).
- Click **Create repository**.

### 3. Push your files
In the VS Code terminal (inside your project folder):
```
git init
git add .
git commit -m "Compact crop disease detector"
git branch -M main
git remote add origin https://github.com/<your-username>/crop-disease-detector.git
git push -u origin main
```
Replace `<your-username>` with your GitHub username. GitHub will ask you to sign in
(or paste a token) the first time.

Refresh your GitHub page — your files are now online.

> **Important:** the `.gitignore` file stops the big `data/` folder and training files
> from being uploaded. Never commit the dataset — it is hundreds of MB.

---

## Part C — Running on the internet (optional)

- **Google Colab** (easiest): https://colab.research.google.com — free, runs in the browser,
  good for training. You can `git clone` your repo there and run the code.
- **GitHub Codespaces**: VS Code in the browser, opened from your repo (has a free monthly limit).
- **GitHub Actions**: for automatic runs (tests, builds) — not needed for a hackathon demo.

For the hackathon, running locally in VS Code (Part A) and showing the demo is enough.

---

## Quick command cheat-sheet

| What you want | Command |
| --- | --- |
| Check Python | `python --version` |
| Activate venv (Windows) | `venv\Scripts\activate` |
| Install packages | `pip install -r requirements.txt` |
| Predict one image | `python predict.py leaf.jpg` |
| Open web demo | `python app.py` |
| Save changes to GitHub | `git add .` then `git commit -m "msg"` then `git push` |
