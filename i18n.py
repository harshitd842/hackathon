"""
i18n.py - Hindi (Devanagari) text for the crop-disease app.

The English advisory lives in farm_tools.py. This module mirrors the same logic
for Hindi: same class names, same treatments, same severity priors, same rupee
arithmetic - only the words change. Import farm_tools for the numbers so the two
languages can never drift apart.

Used by app.py when the language switch is set to हिंदी.
"""
import os
import sqlite3

import farm_tools
from farm_tools import (
    CONFIDENT_ENOUGH_TO_NAME_CROP,
    DB_PATH,
    SIBLING_CROPS,
    calculate_roi,
    severity_for,
)

# ---------------------------------------------------------------- crops
CROPS_HI = {
    "Apple": "सेब",
    "Blueberry": "ब्लूबेरी",
    "Cherry_(including_sour)": "चेरी",
    "Corn_(maize)": "मक्का",
    "Grape": "अंगूर",
    "Orange": "संतरा",
    "Peach": "आड़ू",
    "Pepper,_bell": "शिमला मिर्च",
    "Potato": "आलू",
    "Raspberry": "रसभरी",
    "Soybean": "सोयाबीन",
    "Squash": "कद्दूवर्गीय",
    "Strawberry": "स्ट्रॉबेरी",
    "Tomato": "टमाटर",
}

# ---------------------------------------------------------------- diseases
DISEASES_HI = {
    "Apple_scab": "पपड़ी रोग",
    "Black_rot": "काला सड़न",
    "Cedar_apple_rust": "सीडर-सेब रतुआ",
    "healthy": "स्वस्थ",
    "Powdery_mildew": "चूर्णिल आसिता",
    "Cercospora_leaf_spot Gray_leaf_spot": "ग्रे लीफ स्पॉट",
    "Common_rust_": "सामान्य रतुआ",
    "Northern_Leaf_Blight": "उत्तरी पत्ती झुलसा",
    "Esca_(Black_Measles)": "एस्का (ब्लैक मीज़ल्स)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "पत्ती झुलसा",
    "Haunglongbing_(Citrus_greening)": "हरित रोग (सिट्रस ग्रीनिंग)",
    "Bacterial_spot": "जीवाणु धब्बा",
    "Early_blight": "अगेती झुलसा",
    "Late_blight": "पछेती झुलसा",
    "Leaf_Mold": "पत्ती फफूंद",
    "Septoria_leaf_spot": "सेप्टोरिया पत्ती धब्बा",
    "Spider_mites Two-spotted_spider_mite": "मकड़ी घुन (दो-धब्बा)",
    "Target_Spot": "लक्ष्य धब्बा",
    "Tomato_Yellow_Leaf_Curl_Virus": "पीला पत्ती मरोड़ वायरस",
    "Tomato_mosaic_virus": "मोज़ेक वायरस",
    "Leaf_scorch": "पत्ती झुलसा",
}

# ---------------------------------------------------------------- treatments
TREATMENTS_HI = {
    "Potato___Late_blight": (
        "पछेती झुलसा ठंडे और नम मौसम में तेज़ी से फैलता है। मेटालैक्सिल-एम 4% + मैंकोज़ेब 64% "
        "(रिडोमिल गोल्ड) 2 ग्राम/लीटर छिड़कें, या साइमोक्सानिल 8% + मैंकोज़ेब 64% 2 ग्राम/लीटर। "
        "बीमारी शुरू होने से पहले बचाव के लिए मैंकोज़ेब 75% WP 2 ग्राम/लीटर या कॉपर ऑक्सीक्लोराइड "
        "3 ग्राम/लीटर। रोगग्रस्त पत्तियाँ हटा कर नष्ट करें; ऊपर से सिंचाई न करें। 7-10 दिन में दोहराएँ।",
        "मेटालैक्सिल-एम 4% + मैंकोज़ेब 64%",
    ),
    "Potato___Early_blight": (
        "मैंकोज़ेब 75% WP 2 ग्राम/लीटर छिड़कें; रोगग्रस्त पत्तियाँ हटा कर नष्ट करें।",
        "मैंकोज़ेब 75% WP",
    ),
    "Tomato___Late_blight": (
        "पछेती झुलसा ठंडे और नम मौसम में तेज़ी से फैलता है। मेटालैक्सिल-एम 4% + मैंकोज़ेब 64% "
        "(रिडोमिल गोल्ड) 2 ग्राम/लीटर छिड़कें, या साइमोक्सानिल 8% + मैंकोज़ेब 64% 2 ग्राम/लीटर। "
        "बीमारी शुरू होने से पहले बचाव के लिए मैंकोज़ेब 75% WP 2 ग्राम/लीटर या कॉपर ऑक्सीक्लोराइड "
        "3 ग्राम/लीटर। रोगग्रस्त पत्तियाँ हटा कर नष्ट करें।",
        "मेटालैक्सिल-एम 4% + मैंकोज़ेब 64%",
    ),
    "Tomato___Early_blight": (
        "मैंकोज़ेब 75% WP 2 ग्राम/लीटर या क्लोरोथालोनिल छिड़कें।",
        "मैंकोज़ेब 75% WP",
    ),
    "Tomato___Septoria_leaf_spot": (
        "क्लोरोथालोनिल या मैंकोज़ेब छिड़कें; पत्तियों पर पानी डालने से बचें।",
        "मैंकोज़ेब 75% WP",
    ),
    "Corn_(maize)___Common_rust_": (
        "संक्रमण शुरू होते ही प्रोपिकोनाज़ोल 25% EC 1 मिली/लीटर छिड़कें।",
        "प्रोपिकोनाज़ोल 25% EC",
    ),
    "Grape___Black_rot": (
        "मैंकोज़ेब या माइक्लोबुटानिल छिड़कें; सूखे हुए फल हटा दें।",
        "मैंकोज़ेब 75% WP",
    ),
    "Apple___Apple_scab": (
        "कलियाँ खिलने के समय से कैप्टान या मैंकोज़ेब छिड़कना शुरू करें।",
        "मैंकोज़ेब 75% WP",
    ),
    "Squash___Powdery_mildew": (
        "घुलनशील गंधक या नीम का तेल छिड़कें; हवा का आवागमन बढ़ाएँ।",
        "नीम तेल",
    ),
}

DEFAULT_TREATMENT_HI = (
    "पक्के इलाज के लिए अपने नज़दीकी KVK / कृषि अधिकारी से सलाह लें।",
    "नीम तेल",
)

SIBLING_CROP_HI = {"Tomato": "टमाटर", "Potato": "आलू"}

# ---------------------------------------------------------------- UI strings
UI = {
    "title": "फसल रोग पहचान - पूरी तरह ऑफ़लाइन, इलाज की सलाह के साथ",
    "desc": (
        "पत्ती की फोटो डालें। मॉडल पूरी तरह ऑफ़लाइन चलता है - न इंटरनेट, न क्लाउड, "
        "न कोई खाता। फोटो में कई पत्ते हों तो हर एक जाँचा जाता है; वरना पत्ती अपने आप "
        "काट ली जाती है। जहाँ मॉडल पक्का नहीं होता, वहाँ लीफ गार्ड साफ़ मना कर देता है। "
        "38 रोग-वर्ग, 14 फसलें, 92.2% सटीकता।"
    ),
    "scope": (
        "**सटीकता का दायरा:** 92.2% उन साफ़, अलग-अलग पत्ती की फोटो पर जो प्रयोगशाला "
        "जैसी हों (PlantVillage)। असली खेत की फोटो पर यह कम रहती है - 30.9% "
        "(PlantDoc बेंचमार्क; जो जवाब देता है उन पर 61.4%)। ऐसे में ऐप अंदाज़ा नहीं लगाता, साफ़ मना कर देता है।\n\n"
        "**सहायक फसलें (14):** टमाटर, आलू, मक्का, सेब, अंगूर, आड़ू, शिमला मिर्च, चेरी, "
        "स्ट्रॉबेरी, कद्दूवर्गीय, ब्लूबेरी, रसभरी, सोयाबीन, संतरा। इस सूची से बाहर का "
        "पौधा डालेंगे तो ऐप ‘भरोसा कम’ कहेगा।"
    ),
    "img_label": "पत्ती की फोटो (एक पत्ता, या पूरा पौधा)",
    "acre_label": "खेत का आकार (एकड़)",
    "lang_label": "भाषा / Language",
    "check_btn": "जाँच करें",
    "speak_btn": "🔊 सुनें",
    "out_label": "निदान और सलाह",
    "top3_label": "शीर्ष 3 अनुमान",
    "no_photo": "पहले पत्ती की फोटो डालें।",
    "low_conf": (
        "⚠️ **भरोसा कम है** - मॉडल इस फोटो के बारे में पक्का नहीं है।\n\n"
        "**सबसे संभावित:** {best}  ({conf:.0f}%)\n\n"
        "यह गलत भी हो सकता है। भरोसेमंद नतीजे के लिए **एक पत्ती** की पास से, "
        "साफ़ पृष्ठभूमि और अच्छी रोशनी में फोटो डालें।\n\n"
        "_({reason})_"
    ),
    "healthy_head": "✅ **आपका पौधा स्वस्थ दिख रहा है - किसी इलाज की ज़रूरत नहीं।**",
    "healthy_tail": (
        "नियमित निगरानी रखें; शुरुआत में पहचान ही सबसे अच्छी सुरक्षा है।\n\n"
        "_अगर नए धब्बे या रंग बदलना दिखे, तो नई फोटो लें।_"
    ),
    "spray_yes": "✅ **छिड़काव की सलाह दी जाती है।**",
    "spray_no": "⚠️ इस गंभीरता पर इलाज का खर्च फ़ायदे से ज़्यादा हो सकता है - फसल पर नज़र रखें।",
    "disclaimer": "_यह सलाह सामान्य जानकारी है - अपने नज़दीकी KVK / कृषि अधिकारी से पुष्टि करें।_",
}


# ---------------------------------------------------------------- helpers
def pretty_hi(class_name):
    """'Tomato___Late_blight' -> 'टमाटर - पछेती झुलसा'."""
    if "___" not in class_name:
        return class_name
    crop, disease = class_name.split("___", 1)
    crop_hi = CROPS_HI.get(crop, crop)
    disease_hi = DISEASES_HI.get(disease, disease.replace("_", " "))
    return f"{crop_hi} - {disease_hi}"


def treatment_hi(class_name):
    return TREATMENTS_HI.get(class_name, DEFAULT_TREATMENT_HI)


def _stock_hi(chemical):
    """Hindi version of farm_tools.check_pesticide_stock."""
    if not os.path.exists(DB_PATH):
        return "गोदाम की सूची अभी तैयार नहीं है (python seed_inventory.py चलाएँ)"
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT quantity_available, unit, unit_cost_inr FROM inventory WHERE item_name LIKE ?",
            (f"%{chemical}%",),
        ).fetchone()
    finally:
        conn.close()
    if row and row[0] > 0:
        return f"आपके गोदाम में है: {row[0]} {row[1]} (बाज़ार भाव ₹{row[2]}/{row[1]})"
    if row:
        return f"आपके गोदाम में नहीं है। बाज़ार भाव: ₹{row[2]}/{row[1]}"
    return "आपकी स्थानीय सूची में नहीं मिला"


def _diagnosis_hi(class_name, confidence_pct):
    """Same rule as the English side: only name the crop when we are confident."""
    if "___" not in class_name:
        return pretty_hi(class_name), ""
    crop, disease = class_name.split("___", 1)
    pair = SIBLING_CROPS.get(disease.lower().replace(" ", "_"))
    disease_hi = DISEASES_HI.get(disease, disease.replace("_", " "))

    if pair and confidence_pct < CONFIDENT_ENOUGH_TO_NAME_CROP:
        both = " या ".join(SIBLING_CROP_HI.get(c, c).lower() for c in pair)
        return (
            f"{disease_hi}  -  {confidence_pct:.0f}% विश्वास",
            f"**फसल:** {both} - पत्ते लगभग एक जैसे दिखते हैं और रोग भी एक ही है, "
            f"इसलिए नीचे बताया इलाज दोनों के लिए एक जैसा है।",
        )
    return f"{pretty_hi(class_name)}  -  {confidence_pct:.0f}% विश्वास", ""


def _sibling_note_hi(class_name):
    if "___" not in class_name:
        return ""
    crop, disease = class_name.split("___", 1)
    pair = SIBLING_CROPS.get(disease.lower().replace(" ", "_"))
    if not pair:
        return ""
    others = [c for c in pair if c.lower() != crop.lower()]
    if not others:
        return ""
    disease_hi = DISEASES_HI.get(disease, disease.replace("_", " ")).lower()
    a, b = (SIBLING_CROP_HI.get(c, c) for c in pair)
    return (
        f"⚠️ **फसल पर ध्यान दें:** {disease_hi} {a} और {b} दोनों पर होता है - "
        f"पत्ते लगभग एक जैसे दिखते हैं और रोगाणु भी एक ही है। "
        f"अगर आपका पौधा {SIBLING_CROP_HI.get(others[0], others[0])} है, तो ऊपर बताया इलाज वही रहेगा।"
    )


def build_advice_hi(class_name, confidence_pct, acreage):
    """Hindi advisory card. Same numbers as farm_tools.build_advice."""
    if "healthy" in class_name.lower():
        return "\n".join([
            f"**निदान:** {pretty_hi(class_name)}  -  {confidence_pct:.0f}% विश्वास",
            "",
            UI["healthy_head"],
            "",
            UI["healthy_tail"],
        ])

    treatment, chemical = treatment_hi(class_name)
    # the inventory table stores English product names, so query with that one
    _en_chemical = farm_tools.get_treatment(class_name)[1]
    sev = severity_for(class_name) * 100.0
    stock = _stock_hi(_en_chemical)
    in_stock = stock.startswith("आपके गोदाम में है")
    roi = calculate_roi(acreage, sev, treatment_cost_per_acre=0.0 if in_stock else 350.0)

    diag, crop_line = _diagnosis_hi(class_name, confidence_pct)
    note = "" if crop_line else _sibling_note_hi(class_name)
    spray = UI["spray_yes"] if roi["spray_recommended"] else UI["spray_no"]

    return "\n".join([
        f"**निदान:** {diag}",
        *([crop_line, ""] if crop_line else [""]),
        f"**सुझाया इलाज:** {treatment}",
        f"**सुझाया उत्पाद:** {chemical}",
        *([note, ""] if note else []),
        f"**आपके गोदाम में:** {stock}",
        "",
        f"**हिसाब** ({acreage:g} एकड़, ~{sev:.0f}% मानी गई गंभीरता):",
        f"- बिना इलाज अनुमानित नुकसान: ₹{roi['estimated_loss_inr']:,.0f}",
        f"- इलाज का खर्च: ₹{roi['treatment_cost_inr']:,.0f}",
        f"- **छिड़काव से शुद्ध लाभ: ₹{roi['net_benefit_inr']:,.0f}**",
        "",
        spray,
        "",
        UI["disclaimer"],
    ])
