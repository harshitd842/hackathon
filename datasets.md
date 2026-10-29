# Datasets for Indian crop diseases

A shortlist of public datasets you can train on, what each one covers, and — the
part most people skip — **what its licence actually lets you do**.

Read the licence column before you download. Two of the datasets below carry
conditions that would constrain your project, and one has no clear licence at
all.

---

## The three things to check before using any dataset

1. **Is it offline?** Every dataset here is, once downloaded. Downloading needs
   internet once; training and the app never call anything.
2. **Is there a privacy issue?** These are public research datasets, so there is
   no personal data of yours or of any farmer. Your app collects nothing. The one
   hygiene step: run `python download_data.py strip-exif --dir raw` to remove
   camera and GPS metadata from the images.
3. **What does the licence allow?** This is the one that bites. See the rules
   below.

### Licence rules of thumb

| Licence | What it means for you | Verdict |
| --- | --- | --- |
| **CC0 / Public Domain** | Do anything, no conditions | Safe |
| **CC BY 4.0** | Use freely, credit the authors | Safe |
| **Apache 2.0** | Use freely, keep the notice | Safe |
| **CC BY-SA** | Credit, **and your derivative must use the same licence** | Careful |
| **CC BY-NC** | **Non-commercial only** — no paid product, ever | Avoid for a product |
| **CC BY-NC-SA** | Non-commercial **and** share-alike | Avoid for a product |
| **"Unknown" / "© Original Authors"** | You have no permission in writing | Avoid, or ask the author |

If your plan is ever to sell this, or to keep the model proprietary, then **only
CC0, CC BY 4.0 and Apache 2.0 are usable**. Everything else needs a decision.

---

## Safe to use

These have a clear, permissive licence.

### PlantDoc — real field photos

- **Covers:** 13 species, 27 classes, 2,598 images of real field conditions
- **Why it matters:** this is the only widely used set of *real* field photos,
  and the paper shows a PlantVillage-trained model scoring 13–15% on it
- **Licence:** CC BY 4.0 — attribution only, no share-alike
- **Get it:** https://github.com/pratikkayal/PlantDoc-Dataset
- **Cite:** Singh, Jain, Jain, Kayal, Kumawat & Batra (2020), CoDS-COMAD,
  doi:10.1145/3371158.3371196

### PlantVillage — the lab baseline

- **Covers:** 14 species, 38 classes, ~54,000 images (the augmented Kaggle
  version has ~87,000)
- **Why it matters:** this is what your current model was trained on
- **Licence — check which release you have.** The GitHub repo and the common
  Kaggle mirrors say **CC0 1.0**. But the original 2015 release announcement says
  **CC BY-SA 3.0**, and adds that *algorithms trained on the data fall under the
  same licence*. Those two disagree.
- **Get it:** https://github.com/spMohanty/PlantVillage-Dataset
- **Cite:** Mohanty, Hughes & Salathé (2016), Frontiers in Plant Science 7:1419,
  doi:10.3389/fpls.2016.1419

### Crop leaf disease dataset — wheat, rice, cotton, sugarcane

- **Covers:** 6 crops, 45 classes, 15,809 images — Rice 6 classes, Sugarcane 5,
  Cotton 4, **Wheat 15**, Tomato 11, Corn 4
- **Why it matters:** the fastest single download to widen coverage beyond the
  14 PlantVillage crops
- **Licence:** Apache 2.0
- **Get it:** https://www.kaggle.com/datasets/sriganeshravisankar/crop-leaf-disease-dataset

### Crop Disease Detection Dataset — 12 crops, 47 classes

- **Covers:** Apple, Banana, Bean, Corn, Cotton, Grape, Pepper, Potato, Rice,
  Soybean, Tomato, Wheat — 47 classes
- **Includes:** Cotton bacterial blight, cotton curl virus, cotton fusarium wilt
- **Watch out:** the Rice classes have only ~40 images each, so they will train
  poorly
- **Licence:** CC0 — public domain
- **Get it:** https://www.kaggle.com/datasets/ketangathibandhe/crop-disease-detection-dataset

### Sugarcane Leaf Disease Dataset

- **Covers:** 5 classes — healthy, mosaic, red rot, rust, yellow — 2,569 images
- **Why it matters:** photographed on smartphones in Maharashtra, so it is
  field-like rather than lab-like
- **Licence:** CC0 — public domain
- **Get it:** https://www.kaggle.com/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset

### Rice Leaf Disease Images

- **Covers:** 4 diseases — bacterial blight, blast, brown spot, tungro — 5,932
  images
- **Licence:** CC0 — public domain
- **Get it:** https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image

---

## Detection datasets — boxes around each leaf, not one label per photo

Everything above is a **classification** set: one photo, one label. A detection
set is different. Each leaf in a photo gets a bounding box and a label, so a
single photo can carry several labelled leaves.

That matters for two reasons: these are usually real field photos, and the boxes
tell you *where* the disease is, not only what it is.

### PlantDoc — object detection version

- **Covers:** the same 13 species and 30 classes as the classification version,
  but annotated with bounding boxes — 8,851 labels across 2,569 images
- **Licence:** CC BY 4.0 — attribution only
- **Get it:** https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset
- A Roboflow export of the same data also exists, resized to 416×416

### Plant Disease Detection — YOLOv11x benchmark, 116 classes

- **Covers:** 116 classes across corn, tomato, cassava, apple, grape, potato,
  banana, citrus and more; 23,689 images
- **Composition:** 18,533 lab images across 89 classes (plantwild), plus 5,156
  real field images across 27 classes (FieldPlant)
- **Reported result:** mAP50 of 69.8% with YOLOv11x, 57M parameters
- **Caveat:** 116 classes is not 116 diseases. Several classes are healthy leaves
  or plain species labels — "Cassava Healthy", "Raspberry Leaf", "Celery Leaf"
  all appear among the top-scoring classes
- **Get it:** the JK-TK PlantDiseaseDetection repository on Hugging Face

### Plant disease.v1i.yolov11 — 31 classes

- **Covers:** 31 classes, 2,556 images, 8,807 annotations
- **Get it:** the Kunal Jadhav dataset on the Ultralytics platform

### Should you switch to detection?

Not automatically. Detection means a different model, a different export format
and a much larger file. Your app already locates and crops the leaf with
`leaf_segment.py`, so a classification model plus that step gets you most of the
same result today.

The honest use for these sets is **data**, not architecture. They contain real
field photos and multi-leaf images, which is exactly the gap you are trying to
close. Crop the boxes out and feed them to your existing classifier.

Licences: I have verified **CC BY 4.0** for the PlantDoc detection set. I have
not verified the other two, so check before use.

## Check before you use

These exist and look useful, but the licence will constrain you.

### Five Crop Diseases Dataset — 17 classes

- **Covers:** Corn, Potato, Rice, Wheat, Sugarcane — 13,324 images
- **Licence: CC BY-NC-SA 4.0.** Non-commercial **and** share-alike. A paid
  product is not allowed, and anything you build on it must carry the same
  licence. **Do not build your product on this.**
- https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset

### Crops Disease Dataset (vishesh2395) — wheat 15 classes

- **Covers:** Wheat 15 classes, Rice 4, Potato 3, Corn 4 — a strong wheat set
- **Licence: Unknown.** No licence means no permission. Ask the uploader before
  you use it.
- https://www.kaggle.com/datasets/vishesh2395/crops-disease-dataset

### Plant Disease Dataset (rashidthihan) — 71 classes

- **Covers:** 20+ species, 71 classes, ~28,500 images
- **Licence: a custom restriction** — the page states use is restricted to
  activities that adhere to Shariah. That is not a standard open licence.
  **Avoid.**
- https://www.kaggle.com/datasets/rashidthihan/plant-disease-dataset

### New Plant Diseases Dataset (vipoooool) — 38 classes

- **Covers:** 38 classes, 14 crops, ~87,000 augmented images
- **Licence: "Data files © Original Authors"** — not a licence, just a
  copyright claim. Treat the licence as unresolved; it is an augmented copy of
  PlantVillage, so the PlantVillage question above applies.
- https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

### 20k+ Multi-Class Crop Disease Images

- **Covers:** wheat, maize, cotton, sugarcane, rice
- **Licence:** "open, for research and non-commercial purposes" — so
  **non-commercial**. Not usable in a paid product.
- https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images

---

## Suggested plan

A realistic order, using only the safe sets:

| Step | What | Adds |
| --- | --- | --- |
| 1 | **PlantDoc** | real field photos, 27 classes — the biggest accuracy gain |
| 2 | **Crop leaf disease dataset** (Apache 2.0) | wheat, rice, cotton, sugarcane — 45 classes |
| 3 | **Rice + Sugarcane** CC0 sets | fills out those two crops with more images |
| 4 | **Your own photos** | the only set that matches your actual users |

That takes you from **38 classes across 14 crops** to roughly **70–80 classes
across 20 crops** — while staying on permissive licences.

---

## Before you ship

- [ ] Every dataset you used is listed in your README with its licence and a link
- [ ] You have attributed every CC BY / CC BY-SA dataset
- [ ] No CC BY-NC dataset is anywhere in your pipeline
- [ ] EXIF metadata stripped from every image in `raw/`
- [ ] You can name, for each dataset, why its licence permits your use

If you cannot answer the last one for a dataset, do not ship with it.
