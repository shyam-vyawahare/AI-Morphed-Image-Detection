import cv2
import dlib
import numpy as np
import os

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download this separately
predictor = dlib.shape_predictor(predictor_path)

# Get facial landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) != 1:
        raise Exception("Image must contain exactly one face.")
    
    landmarks = predictor(gray, faces[0])
    return np.array([(p.x, p.y) for p in landmarks.parts()])

# Apply affine transform to triangle
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

# Morph one triangle
def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    # Find bounding rectangles
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points
    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))

    # Crop the regions
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    img_subsection = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
    img_subsection = img_subsection * (1 - mask) + img_rect * mask
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img_subsection

# Perform full image morphing
def morph_faces(img1, img2, alpha=0.5):
    img1 = cv2.resize(img1, (600, 600))
    img2 = cv2.resize(img2, (600, 600))
    points1 = get_landmarks(img1)
    points2 = get_landmarks(img2)

    points = []
    for i in range(len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((int(x), int(y)))

    # Delaunay triangulation
    rect = (0, 0, 600, 600)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indices = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for p in pts:
            for i, point in enumerate(points):
                if abs(p[0] - point[0]) < 1 and abs(p[1] - point[1]) < 1:
                    idx.append(i)
        if len(idx) == 3:
            indices.append(tuple(idx))

    morphed_img = np.zeros(img1.shape, dtype=img1.dtype)

    for i in indices:
        t1 = [points1[i[0]], points1[i[1]], points1[i[2]]]
        t2 = [points2[i[0]], points2[i[1]], points2[i[2]]]
        t = [points[i[0]], points[i[1]], points[i[2]]]
        morph_triangle(img1, img2, morphed_img, t1, t2, t, alpha)

    return morphed_img

# Load images
img1 = cv2.imread("face1.jpg")  # original face 1
img2 = cv2.imread("face2.jpg")  # original face 2

# Create morphed face
result = morph_faces(img1, img2, alpha=0.5)
cv2.imwrite("morphed_face.jpg", result)
cv2.imshow("Morphed", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
