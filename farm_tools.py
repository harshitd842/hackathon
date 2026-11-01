"""
farm_tools.py - the local farm-advisory logic (the "agent tools").

Everything here runs offline on the local machine:
  1. look up a treatment for a diagnosed disease
  2. check whether the farmer already has the chemical in stock (SQLite file)
  3. estimate the cost/benefit of spraying

The same functions are exposed to a local small language model (SLM) by
mcp_farm_server.py, so the model can turn a diagnosis into a real decision.

NOTE: the treatments below are illustrative starting points. For authoritative
guidance, follow ICAR / your State Agriculture Department / local KVK advice.
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "farm_storehouse.db")

# disease class -> (treatment text, suggested product name)
TREATMENTS = {
    "Potato___Late_blight":        ("Late blight spreads fast in cool, humid weather. Spray Metalaxyl-M 4% + Mancozeb 64% (Ridomil Gold) @ 2 g/L, or Cymoxanil 8% + Mancozeb 64% @ 2 g/L. Mancozeb 75% WP @ 2 g/L or Copper Oxychloride @ 3 g/L works as a protectant before infection sets in. Remove and destroy infected leaves; do not overhead-irrigate. Repeat every 7-10 days.", "Metalaxyl-M 4% + Mancozeb 64%"),
    "Potato___Early_blight":       ("Spray Mancozeb 75% WP @ 2 g/L; remove and destroy infected leaves.", "Mancozeb 75% WP"),
    "Tomato___Late_blight":        ("Late blight spreads fast in cool, humid weather. Spray Metalaxyl-M 4% + Mancozeb 64% (Ridomil Gold) @ 2 g/L, or Cymoxanil 8% + Mancozeb 64% @ 2 g/L. Mancozeb 75% WP @ 2 g/L or Copper Oxychloride @ 3 g/L works as a protectant before infection sets in. Remove and destroy infected leaves.", "Metalaxyl-M 4% + Mancozeb 64%"),
    "Tomato___Early_blight":       ("Spray Mancozeb 75% WP @ 2 g/L or Chlorothalonil.", "Mancozeb 75% WP"),
    "Tomato___Septoria_leaf_spot": ("Spray Chlorothalonil or Mancozeb; avoid overhead watering.", "Mancozeb 75% WP"),
    "Corn_(maize)___Common_rust_": ("Spray Propiconazole 25% EC @ 1 ml/L at early infection.", "Propiconazole 25% EC"),
    "Grape___Black_rot":           ("Spray Mancozeb or Myclobutanil; remove mummified berries.", "Mancozeb 75% WP"),
    "Apple___Apple_scab":          ("Spray Captan or Mancozeb starting at bud break.", "Mancozeb 75% WP"),
    "Squash___Powdery_mildew":     ("Spray wettable sulphur or Neem oil; improve air flow.", "Neem Oil Extract"),
}
DEFAULT_TREATMENT = ("Consult your local KVK / agriculture officer for a confirmed treatment plan.", "Neem Oil Extract")

# Diseases that look the same on closely related crops, where the treatment is
# identical because the pathogen is the same. The model has no notion of "crop",
# so it can name the wrong host. Say so instead of hiding it.
SIBLING_CROPS = {
    "late_blight":  ("Tomato", "Potato"),
    "early_blight": ("Tomato", "Potato"),
}


# rough severity priors by disease type (a real severity model is a future upgrade)
SEVERITY = {"late_blight": 0.35, "early_blight": 0.25, "common_rust": 0.20}
DEFAULT_SEVERITY = 0.25


def pretty(name):
    """'Tomato___Late_blight' -> 'Tomato - Late blight'."""
    return name.replace("___", " - ").replace("_", " ")


def get_treatment(class_name):
    return TREATMENTS.get(class_name, DEFAULT_TREATMENT)


def sibling_crop_note(class_name):
    """Warn when the same disease occurs on a sibling crop and looks identical."""
    if "___" not in class_name:
        return ""
    crop, disease = class_name.split("___", 1)
    pair = SIBLING_CROPS.get(disease.lower().replace(" ", "_"))
    if not pair:
        return ""
    others = [c for c in pair if c.lower() != crop.lower()]
    if not others:
        return ""
    pretty_disease = disease.replace("_", " ").lower()
    return (f"\u26a0\ufe0f **Crop note:** {pretty_disease} affects both "
            f"{pair[0].lower()} and {pair[1].lower()} \u2014 the leaves look almost "
            f"identical and it is the same pathogen. If your plant is "
            f"{others[0].lower()}, the treatment above still applies.")


def severity_for(class_name):
    low = class_name.lower()
    for key, val in SEVERITY.items():
        if key in low:
            return val
    return DEFAULT_SEVERITY


def check_pesticide_stock(chemical_name):
    """Query the local shed inventory. Returns a human-readable string."""
    if not os.path.exists(DB_PATH):
        return "(inventory not set up yet - run: python seed_inventory.py)"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT quantity_available, unit, unit_cost_inr FROM inventory WHERE item_name LIKE ?",
        (f"%{chemical_name}%",),
    )
    row = cur.fetchone()
    conn.close()
    if row and row[0] > 0:
        return f"IN STOCK: {row[0]} {row[1]} already in your shed (market price ₹{row[2]}/{row[1]})."
    if row:
        return f"OUT OF STOCK in your shed. Market price: ₹{row[2]}/{row[1]}."
    return "NOT FOUND in your local inventory."


def calculate_roi(acreage, severity_pct, crop_val_per_acre=25000, treatment_cost_per_acre=350):
    """Estimate crop loss if untreated vs the cost of spraying."""
    acreage = max(float(acreage), 0.0)
    loss = (float(severity_pct) / 100.0) * crop_val_per_acre * acreage
    cost = treatment_cost_per_acre * acreage
    net = loss - cost
    return {
        "estimated_loss_inr": round(loss, 2),
        "treatment_cost_inr": round(cost, 2),
        "net_benefit_inr": round(net, 2),
        "spray_recommended": net > 0,
    }


CONFIDENT_ENOUGH_TO_NAME_CROP = 90.0


def diagnosis_text(class_name, confidence_pct):
    """Name the crop only when the model is confident; otherwise say the crop is ambiguous.

    Tomato and potato share late blight and early blight, and the model has no
    notion of "crop". Below 90% confidence on those diseases it is honest to
    name the disease and stop, rather than assert the wrong host.
    """
    if "___" not in class_name:
        return pretty(class_name), ""
    crop, disease = class_name.split("___", 1)
    pair = SIBLING_CROPS.get(disease.lower().replace(" ", "_"))
    disease_pretty = disease.replace("_", " ")

    if pair and confidence_pct < CONFIDENT_ENOUGH_TO_NAME_CROP:
        both = " or ".join(c.lower() for c in pair)
        return (f"{disease_pretty}  \u2014  {confidence_pct:.0f}% confidence",
                f"**Crop:** {both} \u2014 the leaves look almost identical and it is the "
                f"same pathogen, so the treatment below is the same either way.")
    return f"{pretty(class_name)}  \u2014  {confidence_pct:.0f}% confidence", ""


def build_advice(class_name, confidence_pct, acreage):
    """Produce the structured advisory card shown in the app."""
    # Healthy plant -> no treatment, no economics. Just say it's healthy.
    if "healthy" in class_name.lower():
        return "\n".join([
            f"**Diagnosis:** {pretty(class_name)}  -  {confidence_pct:.0f}% confidence",
            "",
            "✅ **Your plant looks healthy - no treatment needed.**",
            "",
            "Keep monitoring regularly; early detection is the best protection.",
            "",
            "_If you notice new spots or discolouration, take a fresh photo._",
        ])

    treatment, chemical = get_treatment(class_name)
    sev = severity_for(class_name) * 100.0
    stock = check_pesticide_stock(chemical)
    in_stock = stock.startswith("IN STOCK")
    roi = calculate_roi(acreage, sev, treatment_cost_per_acre=0.0 if in_stock else 350.0)
    diag, crop_line = diagnosis_text(class_name, confidence_pct)
    # the standalone crop note only adds value when we DID name a crop
    note = "" if crop_line else sibling_crop_note(class_name)

    spray = ("✅ **Spray recommended.**" if roi["spray_recommended"]
             else "⚠️ At this severity, treatment may not be worth the cost yet - monitor the crop.")
    return "\n".join([
        f"**Diagnosis:** {diag}",
        *([crop_line, ""] if crop_line else [""]),
        f"**Recommended treatment:** {treatment}",
        f"**Suggested product:** {chemical}",
        *([note, ""] if note else []),
        f"**Your local stock:** {stock}",
        "",
        f"**Economics** ({acreage:g} acre, ~{sev:.0f}% assumed severity):",
        f"- Est. crop loss if untreated: ₹{roi['estimated_loss_inr']:,.0f}",
        f"- Est. treatment cost: ₹{roi['treatment_cost_inr']:,.0f}",
        f"- **Net benefit if you spray: ₹{roi['net_benefit_inr']:,.0f}**",
        "",
        spray,
        "",
        "_Advisory is illustrative - please confirm with your local KVK / agriculture officer._",
    ])
