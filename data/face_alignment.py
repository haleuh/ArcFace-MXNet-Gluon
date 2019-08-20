import numpy as np
from skimage import transform as trans

lm_prnet = np.array([36, 45, 30, 48, 54], dtype=np.int32)
align_insight = np.array([[38.2946, 51.6963],
                          [73.5318, 51.5014],
                          [56.0252, 71.7366],
                          [41.5493, 92.3655],
                          [70.7299, 92.2041]], dtype=np.float32)
align_default = align_insight - np.array([0, 15])


def get_landmarks(landmarks):
    landmarks = landmarks[lm_prnet, :2]
    return landmarks


def align_face_and_landmarks(img, landmarks, output_shape=(112, 112)):
    lms = get_landmarks(landmarks)
    dst = lms.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(align_default, dst)
    out = trans.warp(img, tform, output_shape=output_shape, preserve_range=True)
    pad_landmarks = np.pad(landmarks, [(0, 0), (0, 1)], 'constant', constant_values=1).transpose()
    new_landmarks = np.dot(np.linalg.inv(tform.params), pad_landmarks).transpose()
    return out.astype(np.uint8), new_landmarks[:, :2].astype(np.float32)


def align_face(img, landmarks, output_shape=(112, 112)):
    lms = get_landmarks(landmarks)
    dst = lms.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(align_default, dst)
    out = trans.warp(img, tform, output_shape=output_shape, preserve_range=True)
    return out.astype(np.uint8)


def align_landmarks(landmarks):
    lms = get_landmarks(landmarks)
    dst = lms.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(align_default, dst)
    pad_landmarks = np.pad(landmarks, [(0, 0), (0, 1)], 'constant', constant_values=1).transpose()
    new_landmarks = np.dot(np.linalg.inv(tform.params), pad_landmarks).transpose()
    return new_landmarks[:, :2].astype(np.float32)
