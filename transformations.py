import cv2
import numpy as np
import os
import argparse
from plantcv import plantcv as pcv

def apply_gaussian_blur(image):
    """Create a high-contrast edge detection similar to Figure IV.2"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Laplacian edge detection to highlight leaf structure
    laplacian = cv2.Laplacian(blurred, cv2.CV_8U, ksize=5)
    # Invert the image to match the example (structures in white)
    result = cv2.bitwise_not(laplacian)
    # Convert back to 3-channel format for consistent return type
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def create_mask(image):
    """Create a mask similar to Figure IV.3 highlighting internal structures"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply threshold to create a binary mask for the leaf outline
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours to identify the leaf
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create mask for the leaf outline
    mask = np.zeros_like(binary)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # Create the result image with original colors where leaf is present
    result = image.copy()
    result[mask == 0] = [0, 0, 0]  # Set background to black
    # Enhance contrast within the leaf to make structures more visible
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.equalizeHist(hsv[:,:,1])  # Enhance saturation channel
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result

def get_roi_objects(image, mask):
    """Create ROI objects visualization similar to Figure IV.4"""
    # Create a binary mask from the input mask if it's not already binary
    if len(mask.shape) == 3:
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    else:
        binary_mask = mask

    # Find contours of the leaf
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create result image
    result = image.copy()

    if contours:
        # Get the largest contour (the leaf)
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw a bright green outline around the leaf
        cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 3)
        # Add a blue rectangle around the entire leaf
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return result

def analyze_object(image):
    """Combine transformations to analyze leaf structure and find centers."""
    # --- Step 1: Preprocessing ---
    result = image.copy()
    blurred = apply_gaussian_blur(result)             # Gaussian blur
    masked = create_mask(result)                      # Apply mask
    roi_img = get_roi_objects(result, masked)         # Draw ROI box
    pseudo_img = extract_pseudolandmarks(result, masked)  # Draw pseudolandmarks
    # --- Step 2: Combine all overlays visually (semi-transparent stacking) ---
    overlay = result.copy()
    overlay = cv2.addWeighted(overlay, 0.5, blurred, 0.5, 0)
    overlay = cv2.addWeighted(overlay, 0.7, masked, 0.3, 0)
    overlay = cv2.addWeighted(overlay, 0.8, roi_img, 0.2, 0)
    overlay = cv2.addWeighted(overlay, 0.9, pseudo_img, 0.1, 0)

    # --- Step 3: Find leaf contour and centers from mask ---
    # Ensure binary
    if len(masked.shape) == 3:
        gray_mask = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    else:
        binary_mask = masked

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(overlay, [largest_contour], -1, (0, 255, 0), 2)
        # Geometric centroid
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(overlay, (cx, cy), 7, (0, 0, 255), -1)
        else:
            cx, cy = -1, -1

        # Medial-axis center
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
        cv2.circle(overlay, max_loc, 7, (0, 255, 255), -1)

        return overlay

    return overlay


def extract_pseudolandmarks(image, mask):
    """Create pseudolandmarks visualization similar to Figure IV.6"""
    # Create a binary mask from the input mask if it's not already binary
    if len(mask.shape) == 3:
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    else:
        binary_mask = mask

    # Find contours of the leaf
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create result image
    result = image.copy()

    if contours:
        # Get the largest contour (the leaf)
        largest_contour = max(contours, key=cv2.contourArea)
        # Generate pseudolandmarks on the contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        # Draw the pseudolandmarks as orange circles
        for point in approx:
            cv2.circle(result, tuple(point[0]), 5, (0, 165, 255), -1)
        # Draw magenta circle at interesting points (like the leaf tip)
        # Find the point furthest from the centroid
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            max_dist = 0
            furthest_point = None

            for point in largest_contour:
                dist = np.sqrt((point[0][0] - cx)**2 + (point[0][1] - cy)**2)

                if dist > max_dist:
                    max_dist = dist
                    furthest_point = tuple(point[0])

            if furthest_point:
                cv2.circle(result, furthest_point, 7, (255, 0, 255), -1)

    return result

def get_color_histogram(image):
    """Calculate and draw color histogram"""
    # Create a mask to only consider the leaf part
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Convert to RGB for better visualization
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Calculate histograms for each channel using the mask
    hist_r = cv2.calcHist([rgb], [0], mask, [256], [0, 256])
    hist_g = cv2.calcHist([rgb], [1], mask, [256], [0, 256])
    hist_b = cv2.calcHist([rgb], [2], mask, [256], [0, 256])
    # Create histogram image
    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
    # Normalize histograms
    cv2.normalize(hist_r, hist_r, 0, 300, cv2.NORM_MINMAX)
    cv2.normalize(hist_g, hist_g, 0, 300, cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, 0, 300, cv2.NORM_MINMAX)

    # Draw histograms
    for i in range(256):
        cv2.line(hist_img, (i, 300), (i, 300-int(hist_r[i])), (255,0,0), 1)
        cv2.line(hist_img, (i, 300), (i, 300-int(hist_g[i])), (0,255,0), 1)
        cv2.line(hist_img, (i, 300), (i, 300-int(hist_b[i])), (0,0,255), 1)

    return hist_img

def process_image(image_path, dst_dir=None, transformation=None):
    """Process a single image with all transformations"""
    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Apply transformations
    analysis_img = analyze_object(image)
    transformations = {
    'original': image,
    'analyze': analysis_img,
    'gaussian': apply_gaussian_blur(image),
    'mask': create_mask(image),
    'roi': get_roi_objects(image, create_mask(image)),
    'landmarks': extract_pseudolandmarks(image, create_mask(image)),
    'RGB saturation': get_color_histogram(image)
}

    if transformation and transformation in transformations:
        # Return single transformation
        return transformations[transformation]

    if dst_dir:
        # Save all transformations
        os.makedirs(dst_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for name, img in transformations.items():
            output_path = os.path.join(dst_dir, f"{base_name}_{name}.jpg")
            cv2.imwrite(output_path, img)
    else:
        # Display all transformations
        for name, img in transformations.items():
            cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Leaf Image Transformation Tool')
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('-src', help='Source directory path')
    parser.add_argument('-dst', help='Destination directory path')
    parser.add_argument('-mask', action='store_true', help='Apply mask transformation only')
    args = parser.parse_args()

    if args.src and args.dst:
        # Process directory

        for filename in os.listdir(args.src):

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.src, filename)
                process_image(image_path, args.dst)
    else:
        # Process single image
        transformation = 'mask' if args.mask else None
        process_image(args.input, None, transformation)

if __name__ == "__main__":
    main()