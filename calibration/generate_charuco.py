import cv2
import yaml
import os

def generate_charuco_board():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    
    # Board parameters
    cols = 5
    rows = 7
    square_size = 30.0  # mm
    marker_size = 22.0  # mm
    dictionary_id = cv2.aruco.DICT_4X4_50
    
    # Use OpenCV 4.8+ new API
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        size=(cols, rows),
        squareLength=square_size,
        markerLength=marker_size,
        dictionary=dictionary
    )
    
    # A4 size is roughly 210 x 297 mm
    # At 300 DPI, 1 inch = 25.4 mm -> 300 pixels = 25.4 mm
    # pixels_per_mm = 300 / 25.4 ≈ 11.81
    # width_px = int(210 * 11.81) = 2480
    # height_px = int(297 * 11.81) = 3507
    
    # Let's just generate a high-res image based on the board dimensions and DPI
    board_width_mm = cols * square_size
    board_height_mm = rows * square_size
    
    dpi = 300
    px_per_mm = dpi / 25.4
    
    # Add a margin
    margin_mm = 15
    image_width = int((board_width_mm + 2 * margin_mm) * px_per_mm)
    image_height = int((board_height_mm + 2 * margin_mm) * px_per_mm)
    
    margin_px = int(margin_mm * px_per_mm)
    
    img = board.generateImage(outSize=(image_width, image_height), marginSize=margin_px)
    
    img_path = os.path.join(out_dir, "charuco_5x7.png")
    cv2.imwrite(img_path, img)
    print(f"ChArUco board image saved to {img_path}")
    print(f"Dimensions: {cols}x{rows} squares")
    print(f"Square size: {square_size}mm, Marker size: {marker_size}mm")
    print("Please print this image at 100% scale (no scaling) on A4 paper.")
    
    # Save parameters to YAML
    params = {
        "board_type": "charuco",
        "dictionary": "DICT_4X4_50",
        "columns": cols,
        "rows": rows,
        "square_size_mm": square_size,
        "marker_size_mm": marker_size
    }
    
    yaml_path = os.path.join(out_dir, "charuco_board_params.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    print(f"Board parameters saved to {yaml_path}")

if __name__ == '__main__':
    generate_charuco_board()
