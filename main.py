from fastapi import FastAPI, UploadFile, File
from PIL import Image
import cv2
import numpy as np
import io

app = FastAPI(title="WriteMatch API")


# --------------------------------------------------
# SKELETONIZATION (Stroke Thinning)
# --------------------------------------------------

def skeletonize(binary_img):
    """
    Reduces thick handwriting strokes to 1-pixel thin skeletons
    so pen thickness differences don't affect similarity.
    """
    skel = np.zeros(binary_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_img.copy()
    
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break
            
    return skel


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(pil_image):
    image = np.array(pil_image.convert("L"))

    max_size = 1200
    height, width = image.shape

    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale))
        )

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Adaptive thresholding for uneven light/shadows from camera photos
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


# --------------------------------------------------
# CROP HANDWRITING
# --------------------------------------------------

def crop_handwriting(binary):
    points = cv2.findNonZero(binary)

    if points is None:
        return binary

    x, y, w, h = cv2.boundingRect(points)
    padding = 20

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(binary.shape[1], x + w + padding)
    y2 = min(binary.shape[0], y + h + padding)

    return binary[y1:y2, x1:x2]


# --------------------------------------------------
# NORMALIZE IMAGE
# --------------------------------------------------

def normalize_image(binary):
    target_size = 600
    height, width = binary.shape

    if height == 0 or width == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    scale = min(
        (target_size - 40) / width,
        (target_size - 40) / height
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        binary,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)

    x = (target_size - new_width) // 2
    y = (target_size - new_height) // 2

    canvas[y:y + new_height, x:x + new_width] = resized

    return canvas


# --------------------------------------------------
# 1. STRUCTURAL SIMILARITY (Grid Density Analysis)
# --------------------------------------------------

def structural_similarity(image_a, image_b, grid_size=8):
    h, w = image_a.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    vec_a = []
    vec_b = []

    for i in range(grid_size):
        for j in range(grid_size):
            cell_a = image_a[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cell_b = image_b[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]

            vec_a.append(np.sum(cell_a > 0))
            vec_b.append(np.sum(cell_b > 0))

    vec_a = np.array(vec_a, dtype=np.float32)
    vec_b = np.array(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return float(np.clip(cosine_sim, 0, 1))


# --------------------------------------------------
# 2. STROKE SIMILARITY (Distance Transform)
# --------------------------------------------------

def stroke_similarity(image_a, image_b):
    skel_a = skeletonize(image_a)
    skel_b = skeletonize(image_b)

    pts_a = np.where(skel_a > 0)
    pts_b = np.where(skel_b > 0)

    if len(pts_a[0]) == 0 or len(pts_b[0]) == 0:
        return 0.0

    # Distance map from Skeleton B
    inv_b = cv2.bitwise_not(skel_b)
    dist_map_b = cv2.distanceTransform(inv_b, cv2.DIST_L2, 5)
    dist_a_to_b = dist_map_b[pts_a]
    avg_dist_a_to_b = np.mean(dist_a_to_b)

    # Distance map from Skeleton A
    inv_a = cv2.bitwise_not(skel_a)
    dist_map_a = cv2.distanceTransform(inv_a, cv2.DIST_L2, 5)
    dist_b_to_a = dist_map_a[pts_b]
    avg_dist_b_to_a = np.mean(dist_b_to_a)

    avg_dist = (avg_dist_a_to_b + avg_dist_b_to_a) / 2.0

    # Convert pixel distance into exponential decay score
    # Score drops gracefully as spatial distance increases
    score = np.exp(-avg_dist / 14.0)
    return float(np.clip(score, 0, 1))


# --------------------------------------------------
# 3. SHAPE SIMILARITY (Multi-Contour Property Comparison)
# --------------------------------------------------

def shape_similarity(image_a, image_b):
    contours_a, _ = cv2.findContours(image_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(image_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_a or not contours_b:
        return 0.0

    # Top 15 largest contours to capture all main letters
    top_a = sorted(contours_a, key=cv2.contourArea, reverse=True)[:15]
    top_b = sorted(contours_b, key=cv2.contourArea, reverse=True)[:15]

    def extract_features(cnts):
        aspect_ratios = []
        solidities = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h > 0:
                aspect_ratios.append(w / float(h))
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidities.append(area / float(hull_area))
        
        avg_ar = np.mean(aspect_ratios) if aspect_ratios else 1.0
        avg_sol = np.mean(solidities) if solidities else 0.5
        return avg_ar, avg_sol

    ar_a, sol_a = extract_features(top_a)
    ar_b, sol_b = extract_features(top_b)

    ar_diff = abs(ar_a - ar_b) / (ar_a + ar_b + 1e-5)
    sol_diff = abs(sol_a - sol_b) / (sol_a + sol_b + 1e-5)

    score = 1.0 - (0.5 * ar_diff + 0.5 * sol_diff)
    return float(np.clip(score, 0, 1))


# --------------------------------------------------
# FINAL COMPARISON LOGIC
# --------------------------------------------------

def calculate_similarity(image_a, image_b):
    binary_a = preprocess_image(image_a)
    binary_b = preprocess_image(image_b)

    cropped_a = crop_handwriting(binary_a)
    cropped_b = crop_handwriting(binary_b)

    normalized_a = normalize_image(cropped_a)
    normalized_b = normalize_image(cropped_b)

    structural_score = structural_similarity(normalized_a, normalized_b)
    stroke_score = stroke_similarity(normalized_a, normalized_b)
    shape_score_val = shape_similarity(normalized_a, normalized_b)

    # Weighted calculation tuned for handwriting
    final_score = (
        structural_score * 0.35 +
        stroke_score * 0.45 +
        shape_score_val * 0.20
    )

    percentage = final_score * 100
    percentage = max(0, min(100, percentage))

    return {
        "similarity": round(percentage, 2),
        "structural_score": round(structural_score * 100, 2),
        "stroke_score": round(stroke_score * 100, 2),
        "shape_score": round(shape_score_val * 100, 2)
    }


# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "WriteMatch backend is running!"}


@app.post("/compare")
async def compare_handwriting(
    sample_a: UploadFile = File(...),
    sample_b: UploadFile = File(...)
):
    try:
        image_a_data = await sample_a.read()
        image_b_data = await sample_b.read()

        image_a = Image.open(io.BytesIO(image_a_data))
        image_b = Image.open(io.BytesIO(image_b_data))

        result = calculate_similarity(image_a, image_b)

        return {
            "success": True,
            "similarity": result["similarity"],
            "structural_score": result["structural_score"],
            "stroke_score": result["stroke_score"],
            "shape_score": result["shape_score"],
            "message": "Handwriting comparison completed"
        }

    except Exception as e:
        return {
            "success": False,
            "similarity": 0,
            "message": f"Comparison failed: {str(e)}"
        }