import cv2
import mediapipe as mp
import numpy as np
import math

LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                            380, 381, 382, 362]  # Left eye landmarks

RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                            144, 163, 7]  # Right eye landmarks

LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]  # Left iris landmarks
RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]  # Right iris landmarks

NOSE_LANDMARKS = [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48,
                        278, 219, 439, 59, 289, 218, 438, 237, 457, 44, 19, 274]  # Nose landmarks

MOUTH_LANDMARKS = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39,
                        37]  # Mouth landmarks

def main():
    cap = cv2.VideoCapture(0)

    frame_width = 1280
    frame_height = 720

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
								        max_num_faces=2)

    rope_texture = cv2.imread("rope.png")
    bg_texture = cv2.imread("torture.jpg")
    resized_bg = cv2.resize(bg_texture, (frame_width, frame_height))
    resized_bg_inv = 255 - resized_bg
    resized_bg_hsv = cv2.cvtColor(resized_bg_inv, cv2.COLOR_BGR2HSV)
    value = 100 #whatever value you want to add
    resized_bg_hsv[:,:,2] = cv2.add(resized_bg_hsv[:,:,2], value)
    resized_bg_dimmed = cv2.cvtColor(resized_bg_hsv, cv2.COLOR_HSV2BGR)

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = filter_face_augument(frame, face_mesh, resized_bg_dimmed, rope_texture)
        cv2.imshow("video", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break            
    cv2.destroyAllWindows()


def filter_face_augument(frame, face_mesh, bg_texture, rope_texture):
    frame_height, frame_width = frame.shape[:2]
    fm_result = face_mesh.process(frame)
        
    landmarks = {}
    
    
    frame_prep = np.ones_like(frame) * 255
    frame_prep = cv2.addWeighted(frame_prep, 0.9, bg_texture, 0.1, 1)
    # frame_prep = (bg_texture * 1.5).astype(np.uint8)
    frame_prep = bg_texture
    
    if fm_result.multi_face_landmarks:
    
        for facial_landmarks in fm_result.multi_face_landmarks:
            
            # Initialize lists in the landmarks dictionary to store each facial feature's coordinates
            landmarks["left_eye_landmarks"] = []
            landmarks["right_eye_landmarks"] = []
            landmarks["left_iris_landmarks"] = []
            landmarks["right_iris_landmarks"] = []
            landmarks["mouth_landmarks"] = []
            landmarks["nose_landmarks"] = []
            landmarks["all_landmarks"] = []  # Store all face landmarks for complete face mesh
            
            
            # Loop through all face landmarks
            for i, lm in enumerate(facial_landmarks.landmark):
                x, y = int(lm.x * frame_width), int(lm.y * frame_height)  # Convert normalized coordinates to pixel values
                # Store the coordinates of all landmarks
                landmarks["all_landmarks"].append((x, y))
                # Store specific feature landmarks based on the predefined indices
                if i in LEFT_EYE_LANDMARKS:
                    landmarks["left_eye_landmarks"].append((x, y))  # Left eye
                if i in RIGHT_EYE_LANDMARKS:
                    landmarks["right_eye_landmarks"].append((x, y))  # Right eye
                if i in LEFT_IRIS_LANDMARKS:
                    landmarks["left_iris_landmarks"].append((x, y))  # Left iris
                if i in RIGHT_IRIS_LANDMARKS:
                    landmarks["right_iris_landmarks"].append((x, y))  # Right iris
                if i in MOUTH_LANDMARKS:
                    landmarks["mouth_landmarks"].append((x, y))  # Mout
                if i in NOSE_LANDMARKS:
                    landmarks["nose_landmarks"].append((x, y))  # Mout


            r_eye_lm_pts = np.array(landmarks["right_eye_landmarks"])
            r_eye_rect = cv2.boundingRect(r_eye_lm_pts) 
            r_eye_scaled, r_eye_mask = scale_feature(frame, r_eye_lm_pts, r_eye_rect)
            
            
            l_eye_lm_pts = np.array(landmarks["left_eye_landmarks"])
            l_eye_rect = cv2.boundingRect(l_eye_lm_pts) 
            l_eye_scaled, l_eye_mask = scale_feature(frame, l_eye_lm_pts, l_eye_rect)

            mouth_lm_pts = np.array(landmarks["mouth_landmarks"])
            mouth_rect = cv2.boundingRect(mouth_lm_pts) 
            mouth_scaled, mouth_mask = scale_feature(frame, mouth_lm_pts, mouth_rect)
            
            face_lm_pts = np.array(landmarks["all_landmarks"])
            face_rect = cv2.boundingRect(face_lm_pts)
            face_pixelated = pixelate_feature(frame, face_lm_pts, face_rect, res_factor=np.random.rand() * 0.7 + 0.3)
            
            face_center_x = face_rect[0] + face_rect[2] // 2
            face_center_y = face_rect[1] + face_rect[3] // 2
            
            rope_center = (face_center_x, int(face_center_y - face_center_y * 0.1))
            rope_thickness = 30

            frame_prep = tile_image_between_points(frame_prep, rope_texture, rope_center, (0, 0), rope_thickness)                
            frame_prep = tile_image_between_points(frame_prep, rope_texture, rope_center, (frame_width, 0), rope_thickness)                
            frame_prep = tile_image_between_points(frame_prep, rope_texture, rope_center, (0, frame_height), rope_thickness)                
            frame_prep = tile_image_between_points(frame_prep, rope_texture, rope_center, (frame_width, frame_height), rope_thickness)                
            
            mouth_new_x, mouth_new_y = align_centers(mouth_rect, mouth_scaled.shape)
            l_eye_new_x, l_eye_new_y = align_centers(l_eye_rect, l_eye_scaled.shape)
            r_eye_new_x, r_eye_new_y = align_centers(r_eye_rect, r_eye_scaled.shape)
            
            frame_prep = overlay_image(frame_prep, face_pixelated, face_rect[0], face_rect[1])
            frame_prep = overlay_image(frame_prep, mouth_scaled, mouth_new_x, mouth_new_y)
            frame_prep = overlay_image(frame_prep, r_eye_scaled, r_eye_new_x, r_eye_new_y)
            frame_prep = overlay_image(frame_prep, l_eye_scaled, l_eye_new_x, l_eye_new_y)               



    frame = frame_prep
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = (frame_gray < 100).astype(np.uint8) * 255
    frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
    
    return frame

def tile_image_between_points(image, template, from_point, to_point, thickness=10):
    # Calculate vector
    dx, dy = to_point[0] - from_point[0], to_point[1] - from_point[1]
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx))    
    
    # print(dx, dy)

    # Resize image to fit desired thickness
    h, w = template.shape[:2]
    scale = thickness / h
    resized_img = cv2.resize(template, (int(w*scale), thickness), interpolation=cv2.INTER_AREA)

    # Calculate number of tiles needed
    num_tiles = int(length / resized_img.shape[1]) + 1

    tiled = np.tile(resized_img, (1, num_tiles, 1))
    th, tw = tiled.shape[:2]
    
    center = (tw // 2, th //2)
    
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    new_width = abs(dx)
    new_height = abs(dy)
    
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
 
    rotated_img = cv2.warpAffine(tiled, rotation_matrix, (new_width, new_height), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

    top_left_x, top_left_y = from_point

    if not (dy > 0 and dx > 0):
        rotated_img = cv2.flip(rotated_img, 0)
    else:
        rotated_img = cv2.flip(rotated_img, 1)
        

    if dy < 0:
        top_left_y -= new_height
        
    if dx < 0:
        top_left_x -= new_width
            

    mask = (cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY) < 255).astype(np.uint8) * 255
    
    masked = cv2.bitwise_and(rotated_img, rotated_img, mask=mask)
    
    result = overlay_image(image, masked, top_left_x, top_left_y)

    return result

def align_centers(rect, shape):
    x, y, w, h = rect 
    new_x = x + w // 2 - shape[1] // 2
    new_y = y + h // 2 - shape[0] // 2
    
    return new_x, new_y

def scale_feature(image, landmark_points, bounding_rect, scale=2):
    x, y, w, h = bounding_rect

    if x < 0 or y < 0:
        return image
    
    cropped = image[y:y + h, x: x + w]
    
    lm_pts_relative = landmark_points.copy()
    lm_pts_relative[:, 0] -= x 
    lm_pts_relative[:, 1]  -= y
    
    convexhull = cv2.convexHull(lm_pts_relative)
    mask = np.zeros_like(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY))
    mask = cv2.fillConvexPoly(mask, convexhull, 255)
    
    
    new_w = w * scale
    new_h = h * scale

    mask_resized = cv2.resize(mask, (new_w, new_h))
    cropped_resized = cv2.resize(cropped, (new_w, new_h))
    
    no_bg = cv2.bitwise_and(cropped_resized, cropped_resized, mask=mask_resized)
    
    return no_bg, mask_resized

def pixelate_feature(image, landmark_points, bounding_rect, res_factor = 0.2):
    x, y, w, h = bounding_rect
    
    cropped = image[y:y + h, x: x + w]
    
    lm_pts_relative = landmark_points.copy()
    lm_pts_relative[:, 0] -= x 
    lm_pts_relative[:, 1]  -= y
    
    convexhull = cv2.convexHull(lm_pts_relative)
    mask = np.zeros_like(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY))
    mask = cv2.fillConvexPoly(mask, convexhull, 255)
    masked = cv2.bitwise_and(cropped, cropped, mask=mask)
    
    small_w = int(w * res_factor)
    small_h = int(h * res_factor)
    temp = cv2.resize(masked, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return pixelated

def overlay_image(imagebg, imagefg, x, y, mask=None):
    imagebg_h, imagebg_w = imagebg.shape[:2]
    imagefg_h, imagefg_w = imagefg.shape[:2]
    
    x_slice_start, y_slice_start = 0, 0
    
    if x + imagefg_w >= imagebg_w:
        difference = x + imagefg_w - imagebg_w
        imagefg_w -= difference 
    elif x < 0:
        x_slice_start = -x
        imagefg_w += x
        x = 0
        
    if y + imagefg_h >= imagebg_h:
        difference = y + imagefg_h - imagebg_h
        imagefg_h -= difference 
    elif y < 0:
        y_slice_start = -y
        imagefg_h += y
        y = 0

    cropped_fg = imagefg[y_slice_start:y_slice_start+imagefg_h, x_slice_start:x_slice_start+imagefg_w]
    
    global_fg = np.zeros(imagebg.shape, np.uint8)
    global_fg[y:y+imagefg_h, x:x+imagefg_w] = cropped_fg
    
    global_fg_grey = cv2.cvtColor(global_fg, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        inv_mask = (global_fg_grey == 0).astype(np.uint8) * 255
    else:
        cropped_mask = mask[y_slice_start:y_slice_start+imagefg_h, x_slice_start:x_slice_start+imagefg_w]
        inv_mask = np.ones(imagebg.shape[:2], np.uint8)
        inv_mask[y:y+imagefg_h, x:x+imagefg_w] = 255 - cropped_mask
    
    masked_bg = cv2.bitwise_and(imagebg, imagebg, mask=inv_mask)
    
    return cv2.add(masked_bg, global_fg)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
    
    
    