from utils import *

class HLoss:
	'''
	Compares two lines by calculating the distance between their ends in the image.
	'''

	def __init__(self):
		'''
		Constructor.
		'''

	def __call__(self, X, Y, H):
		'''
		Calculate the line loss.

		'''

		HX = de_homo_py(torch.matmul(H, homo_py(X).t()).t())
		loss_reproj = torch.mean(torch.norm(Y - HX, dim=1))
		return loss_reproj