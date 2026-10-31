# UI Decisions — PatteVaidya (पत्तेवैद्य)

A short record of the interface choices behind the app, so future contributors
know *why* things look the way they do.

---

## 1. Hindi first, English on request

The app opens in Hindi. The farmer we designed for is more comfortable reading
Devanagari than English, so Hindi is the default and English is one click away
(the language radio sits under the upload box).

Class names, treatment names and the whole advisory are translated in `i18n.py`.
That file is the single source of truth — the UI never hard-codes a Hindi string.

**Trade-off:** the file is long (38 classes × 9 treatments × 2 languages). We
accepted that in exchange for one place to fix a typo.

---

## 2. One screen, two columns

```
+---------------------------+-----------------------------+
|  Leaf photo (upload)      |  Diagnosis + advisory       |
|  Field size (acres)       |  Top-3 predictions          |
|  Language                 |                             |
+---------------------------+-----------------------------+
```

The left column is *input*, the right column is *answer*. Nothing else competes
for attention — no menus, no tabs, no login.

---

## 3. The confidence guard is part of the design

The model is a lab-image model. On a real field photo it is often unsure, and
guessing loudly is worse than saying nothing.

When the guard rejects a prediction (`guard.check()`), the app shows:

> **भरोसा कम है** — मॉडल इस फोटो के बारे में पक्का नहीं है।

…and **no** treatment advice. This is a deliberate UX choice: a confident wrong
answer costs a farmer money, an honest "I don't know" costs nothing.

The threshold is `CONF_THRESHOLD = 0.65` in `app.py`. It was tuned by hand: 0.60
let too many weak answers through, 0.75 rejected correct ones.

---

## 4. Colour

| Element | Colour | Why |
| --- | --- | --- |
| Primary action | deep green | reads as "crop / healthy" |
| Refusal state | amber | warning, not error — the app is not broken |
| Healthy result | green | matches the primary action |
| Diseased result | rust red | distinct from the refusal amber |

No gradients, no shadows, no decorative illustration. The leaf photo is the only
image on the screen.

---

## 5. Numbers are never naked

Every accuracy claim carries its scope. The line above the upload box says:

> 92.2% उन साफ़, अलग-अलग पत्ती की फोटो पर जो प्रयोगशाला जैसी हों (PlantVillage)।
> असली खेत की फोटो पर यह कम रहती है — 30.9% (PlantDoc बेंचमार्क; जो जवाब देता है
> उन पर 61.4%)।

A judge or a farmer should never see "92%" without knowing what it was measured on.

---

## 6. Supported crops are listed, not implied

14 crops are supported. The app names them **before** the farmer uploads, so an
unsupported leaf (e.g. okra) does not feel like a bug — it feels like a known
limit.

---

## 7. No voice button

An earlier build had a "speak the advisory" button. It was removed: it relied on
the browser's built-in Hindi voice, which many Windows machines do not have, so
it worked on the developer's laptop and silently failed on a farmer's phone.

**Rule:** do not ship a feature that only works on the machine it was built on.

---

## 8. Open questions

- Should the top-3 list show percentages, or just rank order? (Percentages can
  look falsely precise on field photos.)
- Is "भरोसा कम" the right phrase, or is something like "फोटो साफ़ नहीं है"
  clearer when the cause is a blurry image rather than an unfamiliar leaf?
- Should the advisory be printable as a one-page PDF for a Kisan Seva Kendra?

---

*Last updated: 31 Oct 2026*
