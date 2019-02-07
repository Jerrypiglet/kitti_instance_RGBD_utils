import os
import sys
sys.path.append(os.getcwd())
from .utils_F import *

class HLoss:
	'''
	Compares two lines by calculating the distance between their ends in the image.
	'''

	def __init__(self):
		'''
		Constructor.
		'''

	def __call__(self, H, X, Y):
		'''
		Calculate the line loss.

		'''

		# HX = de_homo_py(torch.matmul(H, homo_py(X).t()).t())
		# loss_reproj = torch.mean(torch.norm(Y - HX, dim=1))
		# return loss_reproj

		errors = sampson_dist(H, X, Y, if_homo=False)
		return torch.mean(errors)