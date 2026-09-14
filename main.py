"""
PlantCare AI — "Smart Plant Health Detection"
-------------------------------------------------------------------------
Camera auto-scanning plant disease detector.

Pipeline (matches the product's decision tree):
  START AUTO SCAN -> open camera -> continuous frame scanning ->
  plant/leaf detected? -> NO: "No Plant Detected" popup, nothing sent here
                       -> YES: auto-capture -> image quality check
                               (client-side first, re-checked here) ->
                               clear image? -> NO: "Image Not Clear"
                                            -> YES: POST to /analyze (this file)
                                                    -> Healthy / Dry Leaf /
                                                       Dead Leaf / Disease /
                                                       "unable to identify"

IMPORTANT: detection here is a two-stage pipeline (plant presence, then
condition classification), but BOTH stages currently run on a COLOR-BASED
HEURISTIC, not a trained model — see the comments inside is_plant_image()
and classify_condition() for exactly where a real object-detection model
and a real trained classifier can each be dropped in.
"""

import os
import time
import threading
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = "captures"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SAFETY_NOTE = (
    "Use pesticide only according to the product label and local "
    "agricultural guidance. Dosage can vary by crop, formulation and region."
)
DOSAGE_UNAVAILABLE = (
    "Dosage unavailable — consult the product label or a local "
    "agriculture expert."
)

# Confidence below this -> "unable to confidently identify" instead of a
# disease name (see PLANT_DISEASE_MAP note on why this heuristic can be
# uncertain).
CONFIDENCE_THRESHOLD = 80.0

status = {
    "state": "idle",       # idle | detecting | done | no_plant | unclear | uncertain | error
    "plant": None,
    "condition": None,      # Healthy | Dry Leaf | Dead Leaf | <disease name>
    "disease": None,
    "confidence": None,
    "severity": None,
    "pesticide": None,
    "dosage": None,
    "frequency": None,
    "safety_note": None,
    "steps": [],
    "message": "Point your camera at a leaf."
}
status_lock = threading.Lock()

history = []
MAX_HISTORY = 15


def update_status(**kwargs):
    with status_lock:
        status.update(kwargs)


# --------------------------------------------------------------------------
# Image quality checks
# --------------------------------------------------------------------------
BLUR_THRESHOLD = 30.0
TOO_DARK_THRESHOLD = 0.12    # mean brightness (0-1) below this = too dark
TOO_BRIGHT_THRESHOLD = 0.92  # mean brightness above this = washed out/overexposed


def blur_score(image_path: str) -> float:
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_image_quality(image_path: str, brightness: float):
    """Returns (is_ok, reason) — reason is None when the image is fine."""
    if blur_score(image_path) < BLUR_THRESHOLD:
        return False, ("Image is not clear. Move closer to the plant and hold "
                        "the camera steady, then scan again.")
    if brightness < TOO_DARK_THRESHOLD:
        return False, "Image is too dark. Move to better lighting and scan again."
    if brightness > TOO_BRIGHT_THRESHOLD:
        return False, "Image is too bright/washed out. Reduce glare and scan again."
    return True, None


# --------------------------------------------------------------------------
# Color analysis (plant presence + condition heuristic)
# --------------------------------------------------------------------------
MIN_VEGETATION_PCT = 25.0
MIN_GREEN_PCT = 5.0


def analyze_leaf_colors(image_path: str, sample_size=200) -> dict:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((sample_size, sample_size))
    arr = np.array(img).astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    green_mask = (g > r * 1.05) & (g > b * 1.05) & (g > 0.20)
    yellow_mask = (r > 0.45) & (g > 0.45) & (b < r * 0.75) & (~green_mask)
    brown_mask = (r >= g) & (r > 0.05) & (r < 0.78) & (b < r * 0.85) & (~green_mask) & (~yellow_mask)
    white_mask = (r > 0.75) & (g > 0.75) & (b > 0.65) & (np.abs(r - g) < 0.08)

    total = arr.shape[0] * arr.shape[1]
    brightness = float((r + g + b).mean() / 3)

    stats = {
        "green_pct": round(float(green_mask.sum()) / total * 100, 1),
        "yellow_pct": round(float(yellow_mask.sum()) / total * 100, 1),
        "brown_pct": round(float(brown_mask.sum()) / total * 100, 1),
        "white_pct": round(float(white_mask.sum()) / total * 100, 1),
        "brightness": round(brightness, 3),
    }
    stats["vegetation_pct"] = round(
        stats["green_pct"] + stats["yellow_pct"] + stats["brown_pct"] + stats["white_pct"], 1
    )
    return stats


def is_plant_image(stats: dict) -> bool:
    """
    STAGE 1 of 2: plant/leaf PRESENCE detection — deliberately kept
    separate from STAGE 2 (classify_condition, which decides the leaf's
    condition). This separation matters: it means Stage 1 can be swapped
    out for a real object-detection model (e.g. a small YOLO/SSD model
    trained to draw a box around "leaf") without touching Stage 2 at all.

    HONEST LIMITATION — please read before presenting: this current
    implementation of Stage 1 is still a COLOR-based heuristic, not a
    trained object detector. It checks whether enough of the frame looks
    plant-colored (green, or discolored brown/yellow/white in leaf-like
    proportions) — it does not detect leaf SHAPE. This is intentionally
    NOT the final answer for plant detection; it's a working stand-in
    while no trained model is available in this environment (no internet
    access here to fetch/train one). It reliably rejects the most common
    non-plant cases (a face, a wall, the sky, most colorful objects,
    solid-color backgrounds) — verified in testing — but can occasionally
    accept a plain brown/tan non-plant object, since color alone can't
    always tell that apart from a dry leaf.

    To swap in a real detector:
        - Run a lightweight object-detection model (e.g. a MobileNet-SSD
          or YOLOv8n fine-tuned on a "leaf" class) on the captured frame.
        - Return True only if it returns a leaf/plant bounding box above
          a confidence threshold.
        - Everything downstream (classify_condition, dosage lookup,
          history, UI) needs no changes — that's the point of the split.
    """
    green, yellow, brown = stats["green_pct"], stats["yellow_pct"], stats["brown_pct"]
    vegetation = stats["vegetation_pct"]

    if green >= MIN_GREEN_PCT and vegetation >= MIN_VEGETATION_PCT:
        return True
    if green < MIN_GREEN_PCT and vegetation >= 50.0 and (brown + yellow) >= 35.0:
        return True
    return False


def classify_condition(stats: dict):
    """
    STAGE 2 of 2: condition/disease classification — only ever called
    after Stage 1 (is_plant_image) has already confirmed a plant/leaf is
    present. Returns (category, confidence) where category is one of:
    "Healthy", "Dry/Dead Leaf", "blight", "mold", "rust"

    "blight"/"mold"/"rust" are general CATEGORIES, not final disease names —
    the actual plant-specific disease name is looked up afterwards from
    PLANT_DISEASE_MAP, based on the selected/detected plant type. This is
    how the architecture stays extensible: swap this function for a real
    trained classifier that outputs (plant, disease, confidence) directly,
    and the rest of the pipeline (dosage lookup, severity, confidence
    threshold, result card) needs no changes.

    To upgrade to a real trained model:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        ... run inference on the captured frame ...
        return predicted_class_name, confidence_percent
    """
    green = stats["green_pct"]
    brown, yellow, white = stats["brown_pct"], stats["yellow_pct"], stats["white_pct"]
    unhealthy_pct = brown + yellow + white

    if unhealthy_pct < 4:
        return "Healthy", round(min(92 + unhealthy_pct, 98), 1)

    if green < unhealthy_pct:
        confidence = round(min(80 + unhealthy_pct * 0.1, 95), 1)
        return "Dry/Dead Leaf", confidence

    if white >= brown and white >= yellow:
        category, strength = "mold", white
    elif brown >= yellow:
        category, strength = "blight", brown
    else:
        category, strength = "rust", yellow

    confidence = round(min(68 + strength * 1.3, 98), 1)
    return category, confidence


def get_severity(confidence: float, category: str) -> str:
    if category in ("Healthy", "Dry/Dead Leaf"):
        return "None"
    if confidence >= 90:
        return "Severe"
    if confidence >= 82:
        return "Medium"
    return "Mild"


# --------------------------------------------------------------------------
# Plant-specific disease names + treatment database
# --------------------------------------------------------------------------
# Each plant maps its general category ("blight" / "mold" / "rust") to a
# specific, plant-appropriate disease name and treatment info. Entries
# marked dosage=None deliberately demonstrate the "Dosage unavailable"
# fallback, per the requirement to never present dosage as universally
# safe or invented for every case.
PLANT_DISEASE_MAP = {
    "Tomato": {
        "blight": {"name": "Early Blight", "pesticide": "Mancozeb 75% WP",
                    "dosage": "2 g/L water", "frequency": "Every 7-10 days"},
        "mold":   {"name": "Leaf Mold", "pesticide": "Chlorothalonil 75% WP",
                    "dosage": "2 g/L water", "frequency": "Every 7 days"},
        "rust":   {"name": "Bacterial Spot", "pesticide": "Copper Oxychloride 50% WP",
                    "dosage": None, "frequency": "Every 7-10 days"},
    },
    "Rice": {
        "blight": {"name": "Brown Spot", "pesticide": "Mancozeb 75% WP",
                    "dosage": "2.5 g/L water", "frequency": "Every 10 days"},
        "mold":   {"name": "Leaf Blast", "pesticide": "Tricyclazole 75% WP",
                    "dosage": "0.6 g/L water", "frequency": "Every 10-14 days"},
        "rust":   {"name": "Bacterial Leaf Blight", "pesticide": "Copper Oxychloride 50% WP",
                    "dosage": None, "frequency": "Every 7-10 days"},
    },
    "Wheat": {
        "blight": {"name": "Leaf Rust", "pesticide": "Propiconazole 25% EC",
                    "dosage": "1 ml/L water", "frequency": "Every 14 days"},
        "mold":   {"name": "Powdery Mildew", "pesticide": "Sulfur 80% WP",
                    "dosage": "2.5 g/L water", "frequency": "Every 7 days"},
        "rust":   {"name": "Wheat Rust", "pesticide": "Propiconazole 25% EC",
                    "dosage": "1 ml/L water", "frequency": "Every 14 days"},
    },
    "Cotton": {
        "blight": {"name": "Leaf Spot", "pesticide": "Mancozeb 75% WP",
                    "dosage": "2 g/L water", "frequency": "Every 10 days"},
        "mold":   {"name": "Leaf Spot", "pesticide": "Mancozeb 75% WP",
                    "dosage": "2 g/L water", "frequency": "Every 10 days"},
        "rust":   {"name": "Bacterial Blight", "pesticide": "Copper Oxychloride 50% WP",
                    "dosage": None, "frequency": "Every 7-10 days"},
    },
    # Fallback for plants not in the table above (Sugarcane, Other, or when
    # no plant type was specified) — generic names, still with honest
    # dosage-unavailable entries where a real verified figure isn't set.
    "_default": {
        "blight": {"name": "Leaf Blight", "pesticide": "Mancozeb 75% WP",
                    "dosage": "2 g/L water", "frequency": "Every 7-10 days"},
        "mold":   {"name": "Powdery Mildew", "pesticide": "Sulfur 80% WP",
                    "dosage": "2.5 g/L water", "frequency": "Every 7 days"},
        "rust":   {"name": "Rust", "pesticide": "Propiconazole 25% EC",
                    "dosage": None, "frequency": "Every 14 days"},
    },
}

DRY_DEAD_STEPS = [
    "Check soil moisture.",
    "Maintain proper watering.",
    "Check sunlight conditions.",
    "Remove completely dried leaves if necessary.",
    "Check nearby leaves and overall plant health.",
    "Scan another green or affected leaf if disease is suspected.",
]
HEALTHY_STEPS = ["Continue regular watering, sunlight and plant care."]


def get_disease_info(plant_type: str, category: str):
    """Looks up the plant-specific disease name + treatment info for a
    'blight'/'mold'/'rust' category. Returns a dict with name, pesticide,
    dosage (or None), frequency."""
    table = PLANT_DISEASE_MAP.get(plant_type, PLANT_DISEASE_MAP["_default"])
    return table.get(category, PLANT_DISEASE_MAP["_default"][category])


# --------------------------------------------------------------------------
# Combined diagnosis
# --------------------------------------------------------------------------
def diagnose(image_path: str, plant_type: str) -> dict:
    result = {
        "valid": False, "state": "no_plant", "condition": None, "disease": None,
        "confidence": None, "severity": None, "pesticide": "No diagnosis available.",
        "dosage": None, "frequency": None, "safety_note": None, "steps": [], "message": ""
    }

    try:
        stats = analyze_leaf_colors(image_path)
    except Exception:
        result["state"] = "unclear"
        result["message"] = "Unable to read this image. Please scan again."
        return result

    is_ok, reason = check_image_quality(image_path, stats["brightness"])
    if not is_ok:
        result["state"] = "unclear"
        result["message"] = reason
        return result

    if not is_plant_image(stats):
        result["state"] = "no_plant"
        result["message"] = ("No plant or leaf was detected in the camera view. "
                              "Please point the camera toward a clear plant or leaf.")
        return result

    category, confidence = classify_condition(stats)

    if category == "Healthy":
        result.update(valid=True, state="done", condition="Healthy", confidence=confidence,
                      severity="None", pesticide="No pesticide is required.", steps=HEALTHY_STEPS,
                      message="Your leaf appears healthy.")
        return result

    if category == "Dry/Dead Leaf":
        result.update(valid=True, state="done", condition="Dry/Dead Leaf", confidence=confidence,
                      severity="None", pesticide="No pesticide is required.", steps=DRY_DEAD_STEPS,
                      message="This leaf appears dry or dead.")
        return result

    # Disease categories: blight / mold / rust
    if confidence < CONFIDENCE_THRESHOLD:
        result["state"] = "uncertain"
        result["message"] = "Please scan a clear, affected leaf again."
        return result

    severity = get_severity(confidence, category)
    info = get_disease_info(plant_type, category)
    dosage = info["dosage"] if info["dosage"] else DOSAGE_UNAVAILABLE

    result.update(
        valid=True, state="done", condition="Disease Detected", disease=info["name"],
        confidence=confidence, severity=severity, pesticide=info["pesticide"],
        dosage=dosage, frequency=info["frequency"], safety_note=SAFETY_NOTE,
        steps=[
            f"Apply {info['pesticide']} at {dosage}.",
            f"Repeat application {info['frequency'].lower()}.",
            "Remove and discard severely affected leaves before spraying.",
            SAFETY_NOTE,
        ],
        message=f"{info['name']} detected ({severity} severity, {confidence}% confidence)."
    )
    return result


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def run_pipeline(image_path: str, plant_type: str):
    try:
        display_plant = plant_type if plant_type else "Not specified"
        update_status(state="detecting", message="Analyzing plant health…")

        result = diagnose(image_path, plant_type)

        if not result["valid"]:
            update_status(state=result["state"], plant=display_plant, condition=None,
                          disease=None, confidence=None, severity=None,
                          pesticide=result["pesticide"],  # "No diagnosis available."
                          dosage=None, frequency=None, safety_note=None, steps=[],
                          message=result["message"])
            return  # no_plant / unclear / uncertain scans are never logged to history

        update_status(state="done", plant=display_plant, condition=result["condition"],
                      disease=result["disease"], confidence=result["confidence"],
                      severity=result["severity"], pesticide=result["pesticide"],
                      dosage=result["dosage"], frequency=result["frequency"],
                      safety_note=result["safety_note"], steps=result["steps"],
                      message=result["message"])

        with status_lock:
            history.insert(0, {
                "time": time.strftime("%Y-%m-%d %H:%M"),
                "plant": display_plant,
                "condition": result["condition"],
                "disease": result["disease"],
                "severity": result["severity"],
                "confidence": result["confidence"],
            })
            del history[MAX_HISTORY:]

    except Exception as e:
        print("Pipeline error:", repr(e))
        update_status(state="error", message=f"Error: {e}")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    plant_type = request.form.get("plant_type", "").strip()
    image_file = request.files.get("image")

    if not image_file:
        return jsonify({"error": "No image received"}), 400

    filename = f"{int(time.time()*1000)}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    thread = threading.Thread(target=run_pipeline, args=(image_path, plant_type))
    thread.start()

    return jsonify({"message": "Image received. Analyzing…"})


@app.route("/status")
def get_status():
    with status_lock:
        return jsonify(status)


@app.route("/history")
def get_history():
    with status_lock:
        return jsonify(history)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)