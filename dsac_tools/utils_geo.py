import torch
import numpy as np
import cv2
import math

def rot_to_angle(R):
    return rot12_to_angle_error(np.eye(3, R))

def rot12_to_angle_error(R0, R1):
    r, _ = cv2.Rodrigues(R0.dot(R1.T))
    rotation_error_from_identity = np.linalg.norm(r) / np.pi * 180.
    return rotation_error_from_identity

def dotproduct(v1, v2):
	return sum((a*b) for a, b in zip(v1, v2))

def length(v):
	return math.sqrt(dotproduct(v, v)) + 1e-10

def vector_angle(v1, v2):
	return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))) / np.pi * 180.

def dotproducts(v1s, v2s):
	return np.sum(v1s * v2s, axis=1, keepdims=True)

def vectors_angle(v1s, v2s):
	""" v1s, v2s: [N, 3]
		return: angles in degree: [N, 1] """
	dot_v1sv2s = dotproducts(v1s, v2s)
	length_v1s = np.sqrt(dotproducts(v1s, v1s))
	length_v2s = np.sqrt(dotproducts(v2s, v2s))
	return np.arccos(dot_v1sv2s / (length_v1s * length_v2s)) / np.pi * 180.
