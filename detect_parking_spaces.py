import cv2 as cv
import os
import numpy as np
import time
import math

def detect_parking_spaces(image, min_area=500, max_area=30000, aspect_ratio_range=(1.0, 8.0)):
    
    height, width = image.shape[:2]
    
    roi_y = int(height * 0.6)
    roi_height = height - roi_y
    roi = image[roi_y:, :]
    
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    thresh1 = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv.THRESH_BINARY_INV, 19, 2)
    
    edges = cv.Canny(blurred, 30, 150)
    
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    white_mask = cv.inRange(hsv, lower_white, upper_white)
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    
    color_mask = cv.bitwise_or(white_mask, yellow_mask)
    
    combined_mask = cv.bitwise_or(thresh1, edges)
    combined_mask = cv.bitwise_or(combined_mask, color_mask)
    
    kernel = np.ones((3, 3), np.uint8)
    morph = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel, iterations=1)
    
    kernel_large = np.ones((5, 5), np.uint8)
    morph_enhanced = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel_large, iterations=3)
    morph_enhanced = cv.morphologyEx(morph_enhanced, cv.MORPH_OPEN, kernel_large, iterations=1)
    
    morph = cv.bitwise_or(morph, morph_enhanced)
    
    contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    lines = cv.HoughLinesP(morph, 1, np.pi/180, threshold=25, minLineLength=25, maxLineGap=25)
    
    hough_lines = cv.HoughLines(morph, 1, np.pi/180, threshold=50)
    
    if hough_lines is not None:
        if lines is None:
            lines = []
        else:
            lines = lines.tolist()
            
        for line in hough_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 100 * (-b))
            y1 = int(y0 + 100 * (a))
            x2 = int(x0 - 100 * (-b))
            y2 = int(y0 - 100 * (a))
            lines.append([[x1, y1, x2, y2]])
        
        lines = np.array(lines)
    
    polar_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
            theta = theta % 180
            rho = x1 * math.cos(math.radians(theta)) + y1 * math.sin(math.radians(theta))
            polar_lines.append({'rho': rho, 'theta': theta, 'length': length, 'points': (x1, y1, x2, y2)})
    
    angle_tolerance = 10
    parallel_line_groups = []
    processed_lines = set()
    
    for i, line1 in enumerate(polar_lines):
        if i in processed_lines:
            continue
            
        group = [line1]
        processed_lines.add(i)
        
        for j, line2 in enumerate(polar_lines):
            if j in processed_lines or i == j:
                continue
                
            angle_diff = abs(line1['theta'] - line2['theta'])
            angle_diff = min(angle_diff, 180 - angle_diff)
            
            if angle_diff <= angle_tolerance:
                group.append(line2)
                processed_lines.add(j)
                
        if len(group) >= 2:
            parallel_line_groups.append(group)
    
    parking_spaces = []
    for contour in contours:
        area = cv.contourArea(contour)
        
        if min_area <= area <= max_area:
            x, y, w, h = cv.boundingRect(contour)
            
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                epsilon = 0.04 * cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, epsilon, True)
                
                if 3 <= len(approx) <= 6:

                    adjusted_contour = contour.copy()
                    adjusted_contour[:,:,1] += roi_y

                    contour_lines = []
                    for i in range(len(approx)):
                        pt1 = approx[i][0]
                        pt2 = approx[(i+1) % len(approx)][0]
                        theta = math.degrees(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])) % 180
                        contour_lines.append({'theta': theta, 'points': (pt1[0], pt1[1], pt2[0], pt2[1])})
                    
                    parallelism_score = 0
                    aligned_groups = []
                    
                    for group in parallel_line_groups:
                        group_theta = sum(line['theta'] for line in group) / len(group)
                        
                        for contour_line in contour_lines:
                            angle_diff = abs(contour_line['theta'] - group_theta)
                            angle_diff = min(angle_diff, 180 - angle_diff)
                            
                            if angle_diff <= 15:
                                parallelism_score += 1
                                if group not in aligned_groups:
                                    aligned_groups.append(group)
                    

                    adjusted_contour = contour.copy()
                    adjusted_contour[:,:,1] += roi_y
                    
                    parking_spaces.append({
                        'contour': adjusted_contour,
                        'bbox': (x, y + roi_y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'approx': len(approx),
                        'parallelism_score': parallelism_score,
                        'aligned_groups': len(aligned_groups),
                        'car_adjacent': False,
                        'low_curvature': False
                    })
    
    
    for i, space in enumerate(parking_spaces):
        x, y, w, h = space['bbox']
        ext_x = max(0, x - 20)
        ext_y = max(0, y - 20)
        ext_w = min(width - ext_x, w + 40)
        ext_h = min(height - ext_y, h + 40)
        
        roi_ext = image[ext_y:ext_y+ext_h, ext_x:ext_x+ext_w]
        if roi_ext.size == 0:
            continue
            
        roi_gray = cv.cvtColor(roi_ext, cv.COLOR_BGR2GRAY)
        gradient_x = cv.Sobel(roi_gray, cv.CV_64F, 1, 0, ksize=3)
        gradient_y = cv.Sobel(roi_gray, cv.CV_64F, 0, 1, ksize=3)
        gradient_mag = cv.magnitude(gradient_x, gradient_y)
        
        mean_gradient = np.mean(gradient_mag)
        
        if mean_gradient > 15:

            if space['parallelism_score'] < 4:
                space['parallelism_score'] += 3
                space['car_adjacent'] = True
                
                contour = space['contour']
                if len(contour) >= 5:
                    try:
                        ellipse = cv.fitEllipse(contour)
                        (_, _), (major_axis, minor_axis), _ = ellipse
                        axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
                        
                        if 0.7 <= axis_ratio <= 1.0:
                            space['low_curvature'] = True
                            space['parallelism_score'] += 1
                        else:
                            space['low_curvature'] = False
                    except:
                        space['low_curvature'] = False
                else:
                    space['low_curvature'] = False
            else:
                space['car_adjacent'] = False
        else:
            space['car_adjacent'] = False
    
    parking_spaces.sort(key=lambda x: (x['parallelism_score'], x['aligned_groups']), reverse=True)
    
    return parking_spaces

def process_frame(image_path, output_path=None, display=False, show_debug=True):
    """
    Process a single frame to detect parking spaces.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (optional)
        display: Whether to display the result
        show_debug: Whether to show debug visualizations
        
    Returns:
        Processed image with detected parking spaces and count of free spaces
    """
    image = cv.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    result_image = image.copy()
    
    height, width = image.shape[:2]
    roi_y = int(height * 0.6)
    
    cv.line(result_image, (0, roi_y), (width, roi_y), (255, 0, 0), 2)
    
    parking_spaces = detect_parking_spaces(image)
    
    free_parking_spaces = 0
    
    for i, space in enumerate(parking_spaces):
        x, y, w, h = space['bbox']
        
        color = (0, 255 - i * 30 % 255, i * 40 % 255)

        is_free = not space.get('car_adjacent', False)
        
        if is_free:
            free_parking_spaces += 1
        
        green_color = (0, 255, 0)
        cv.drawContours(result_image, [space['contour']], 0, green_color, 2)
    
    h, w = image.shape[:2]
    combined = np.zeros((h, w*2, 3), dtype=np.uint8)
    combined[:, :w] = image
    combined[:, w:] = result_image
    
    cv.putText(combined, "Original Image", (10, 30), 
              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(combined, f"Detected Spaces: {len(parking_spaces)} | Free: {free_parking_spaces}", (w+10, 30), 
              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if display:
        cv.imshow("Parking Space Detection", combined)
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            cv.destroyAllWindows()
            return None
    
    if output_path:
        cv.imwrite(output_path, combined)
    
    return combined

def process_all_frames(dataset_path, output_dir, display=True, delay=0.1):

    image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')])
    
    if not image_files:
        print(f"No PNG images found in {dataset_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(image_files)} frames...")
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(dataset_path, image_file)
        output_path = os.path.join(output_dir, f"parking_spaces_{i:03d}.png")
        
        print(f"Processing frame {i+1}/{len(image_files)}: {image_file}")
        result = process_frame(image_path, output_path, display)
        
        if result is None:
            print("Processing stopped by user.")
            break
        
        if display and delay > 0:
            time.sleep(delay)
    
    print(f"All frames processed. Results saved to {output_dir}")
    
    if display:
        cv.destroyAllWindows()

def main():
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    process_all_frames(dataset_path, output_dir, display=True, delay=0.1)

if __name__ == "__main__":
    main()
