import cv2
import numpy as np

def end_eff_mask(img):
    """
    Removes:
    1. End-effector polygon (inside region)
    2. White background

    Returns:
        result: cleaned image
        final_mask: mask used
    """

    # -------------------------
    # 1. POLYGON MASK (remove inside)
    # -------------------------
    pts = np.array([
        [297, 1023],
        [370, 822],
        [493, 810],
        [498, 776],
        [532, 774],
        [537, 628],
        [608, 626],
        [614, 778],
        [653, 778],
        [658, 813],
        [770, 822],
        [861, 1023]
    ], dtype=np.int32)

    poly_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts], 255)

    inv_poly_mask = cv2.bitwise_not(poly_mask)

    # -------------------------
    # 2. WHITE BACKGROUND MASK
    # -------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # invert → keep non-white
    non_white_mask = cv2.bitwise_not(white_mask)

    # -------------------------
    # 3. COMBINE MASKS
    # -------------------------
    final_mask = cv2.bitwise_and(inv_poly_mask, non_white_mask)

    # -------------------------
    # 4. APPLY
    # -------------------------
    result = cv2.bitwise_and(img, img, mask=final_mask)

    return result, final_mask


img = cv2.imread("ref_board_90.png")

result, mask = end_eff_mask(img)

# cv2.imshow("Mask", mask)
# cv2.imshow("Cleaned Image", result)
# Find contours from mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get largest contour (board)
cnt = max(contours, key=cv2.contourArea)

# Axis-aligned rectangle
x, y, w, h = cv2.boundingRect(cnt)

# Draw
rect_img = img.copy()
cv2.rectangle(rect_img, (x, y), (x+w, y+h), (0,255,0), 2)

cv2.imshow("Bounding Box", rect_img)

cv2.waitKey(0)
cv2.destroyAllWindows()