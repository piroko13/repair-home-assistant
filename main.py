import cv2
import numpy as np
import math

image = cv2.imread("attached_assets/screws1.png")
if image is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale and apply a stronger blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Detect edges
edges = cv2.Canny(blurred, 30, 100)

# Dilate the edges to merge screw segments
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

detected = 0

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h != 0 else 0
    area = cv2.contourArea(cnt)
    diagonal = math.sqrt(w ** 2 + h ** 2)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0

    #Brightness filter
    mean_val = cv2.mean(gray[y:y + h, x:x + w])[0]

    #Skip if's too dark
    if mean_val < 60:
        continue

    # Draw all contours in blue for reference
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Debug line (optional)
    print(f"Valid screw? Area: {area: .1f}, AR: {aspect_ratio: .2f}, D: {diagonal: .1f}, W: {w}, H: {h}")

    # Smarter screw shape filter
    if 400 < area < 50000 and 0.4 < aspect_ratio < 1.3 and diagonal > 35 and h >= w * 0.75 and h + w > 60 and 0.10 < extent < 0.9:
        detected += 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image, f"Screw {detected}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.imwrite("output_result.png", image)
print(f"Detection complete. {detected} screws found.")
