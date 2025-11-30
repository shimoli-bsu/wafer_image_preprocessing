import numpy as np
import cv2

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes):
    # ----------------------------------------------------------
    # Reshape raw_data → H x W x C
    # Edge Impulse sends flattened pixel data in a single axis.
    # Here: axes = [pixel], so raw_data is already 1 channel.
    # If RGB, axes = [R,G,B]
    # ----------------------------------------------------------
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    # ------------------------------
    # 1. Convert to grayscale
    # ------------------------------
    if len(axes) == 3:
        # assume RGB encoded as [R,G,B]
        img = raw_data.astype(np.uint8)
        img = img.reshape(-1, 3)
        # grayscale conversion
        img = np.dot(img, [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        # already single channel
        img = raw_data.astype(np.uint8).flatten()

    # Infer image size (EI normally gives this)
    # You may hardcode if needed: width = 224, height = 224
    side_len = int(np.sqrt(len(img)))
    img = img.reshape(side_len, side_len)

    # ------------------------------
    # 2. Contrast normalization (CLAHE-like)
    # ------------------------------
    img_norm = cv2.normalize(img, None, alpha=0, beta=255, 
                             norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_norm)

    # ------------------------------
    # 3. Edge enhancement (Sobel + blending)
    # ------------------------------
    sobelx = cv2.Sobel(img_clahe, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_clahe, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)

    # Normalize edge map to [0,255]
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Blend: weighted sum
    # 0.7 * contrast_normalized_image + 0.3 * edges
    enhanced = (0.7 * img_clahe + 0.3 * edges).astype(np.uint8)

    # ------------------------------
    # 4. Scale to [0,1] or [0,255] depending on EI model expectation
    # ------------------------------
    enhanced = enhanced.astype(np.float32)
    enhanced = enhanced / 255.0 * scale_axes

    # ------------------------------
    # 5. Flatten
    # ------------------------------
    features = enhanced.flatten().tolist()

    # No graphs returned, but EI supports it
    graphs = []

    return {
        'features': features,
        'graphs': graphs,
        'fft_used': [],
        'output_config': {
            'type': 'flat',
            'shape': {
                'width': len(features)
            }
        }
    }
