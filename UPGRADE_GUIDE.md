# Upgrade Guide — add the Leaf Guard + local advisory agent

This guide takes you from your current app to the upgraded version, then pushes
it to GitHub. Follow it top to bottom. Commands are for **Windows / PowerShell**.

---

## What you are adding (and why)

1. **Leaf Guard** (`guard.py`) — fixes the biggest visible problem: the model
   guessing "Pepper / Litchi" on photos that aren't one of its 38 crops.
   It reads the model's own confidence and entropy, plus a quick image
   flatness check, and refuses to answer when the input looks wrong.
2. **Local advisory agent** (`farm_tools.py`, `seed_inventory.py`,
   `mcp_farm_server.py`) — turns "you have Late Blight" into a real decision:
   *"you already have Copper Oxychloride in your shed — spray it, save ₹4,200,
   spend ₹0."* This is what makes the project stand out in a hackathon.
3. **Auto-crop + multi-leaf** (`leaf_segment.py`) — the classifier wants a
   single, close-up leaf. This module finds the leaf region(s) in a photo, so
   the app auto-crops a single leaf, and if the photo has **several leaves it
   checks each one** and reports them separately. If the model is unsure about
   a real (compound-leaf) photo, the app also tries a grid of sliding windows
   and keeps the most confident view.

Everything stays **100% offline**.

---

## Step 1 — Add the new files to your project folder

Put these files in the **same folder** as `app.py` (the inner `hackathon-main`):

| New file | What it does |
| --- | --- |
| `guard.py` | Leaf Guard (confidence / entropy check) |
| `leaf_segment.py` | leaf-region detection — auto-crop + multi-leaf support |
| `farm_tools.py` | treatment lookup + inventory check + ROI |
| `seed_inventory.py` | creates the local `farm_storehouse.db` |
| `mcp_farm_server.py` | exposes the tools to a local SLM via MCP |

And **replace** your existing:
- `app.py`  (now includes the Leaf Guard + advisory card)
- `requirements.txt`  (adds nothing new for the core app; `mcp` is optional)

**In VS Code:** drag the files into the project folder, or copy-paste them in.
If VS Code asks "file already exists — overwrite?", choose **Overwrite** for
`app.py` and `requirements.txt`.

---

## Step 2 — Open the terminal and activate the environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
venv\Scripts\activate
```
You should see `(venv)` at the start of the line.

---

## Step 3 — Create the local inventory database

```powershell
python seed_inventory.py
```
You should see: `farm_storehouse.db ready ... with 4 items.`

---

## Step 4 — Run the upgraded app

```powershell
python app.py
```
Wait for: `Running on local URL:  http://127.0.0.1:7860`
Then open that URL in your browser.

**Test it two ways:**
- Upload a **sample leaf** (e.g. `sample_tomato_late_blight.jpg`) → you should
  get a full advisory card with stock + economics.
- Upload a **random photo** (anything that isn't a supported leaf) → the Leaf
  Guard should say *"Not confident — please upload a single leaf."* ✅

> Keep the terminal running while you test. Press `Ctrl + C` only when finished.

---

## Step 5 — (Optional) run the MCP agent server

This is the "agentic" layer for the judges. Only do this if you have time.

```powershell
pip install mcp
python mcp_farm_server.py
```
It runs the tools as an MCP server on localhost. A local SLM (e.g. Qwen2.5-3B via
Ollama) can then call `check_pesticide_stock` and `calculate_treatment_roi`.

> If a 3B model is too slow on your laptop, skip this for the live demo and just
> show the advisory card from Step 4 — that already demonstrates the same logic.

---

## Step 6 — Push everything to GitHub

In the VS Code terminal (inside the project folder):

```powershell
git status
git add .
git commit -m "Add Leaf Guard and local advisory agent"
git push
```

Refresh your GitHub page — the new files (`guard.py`, `farm_tools.py`,
`seed_inventory.py`, `mcp_farm_server.py`) and the updated `app.py` should be
visible.

> **Note:** `farm_storehouse.db` is a local data file. It is fine to commit it so
> the demo works out of the box, OR add it to `.gitignore` and tell people to run
> `python seed_inventory.py`. Either is acceptable.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'numpy'` | `pip install onnxruntime numpy pillow gradio` |
| `can't open file 'app.py'` | you are in the wrong folder — `cd hackathon-main` |
| `running scripts is disabled` | run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first |
| Leaf Guard rejects a *good* photo | lower `CONF_THRESHOLD` in `app.py` (e.g. `0.50`) |
| Leaf Guard accepts a *bad* photo | raise `CONF_THRESHOLD` (e.g. `0.70`) |

---

## How to talk about it (one line for judges)

> "We don't just name a disease — our Leaf Guard refuses to guess when it isn't
> sure, and a local agent checks the farmer's own inventory to tell them whether
> spraying is worth it. All of it runs offline, on the phone, with no cloud."
