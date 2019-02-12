import torch
import numpy as np
import cv2

def rot_to_angle(R):
    return rot12_to_angle_error(np.eye(3, R))

def rot12_to_angle_error(R0, R1):
    r, _ = cv2.Rodrigues(R0.dot(R1.T))
    rotation_error_from_identity = np.linalg.norm(r)/np.pi*180.
    return rotation_error_from_identity