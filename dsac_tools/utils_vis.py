import cv2
import matplotlib.pyplot as plt
import numpy as np
import dsac_tools.utils_misc as utils_misc

def drawlines(img1,img2,lines,pts1,pts2):
    ''' OpenCV function for img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),2,color,2)
        img2 = cv2.circle(img2,tuple(pt2),2,color,2)
    return img1,img2

def scatter_xy(xy, c, im_shape, title='', new_figure=True):
    if new_figure:
        plt.figure(figsize=(30, 8))
    plt.scatter(xy[:, 0], xy[:, 1], s=2, c=c, cmap='rainbow')
    plt.colorbar()
    plt.xlim(0, im_shape[1]-1)
    plt.ylim(im_shape[0]-1, 0)
    plt.title(title)
    plt.show()
    val_inds = utils_misc.within(xy[:, 0], xy[:, 1], im_shape[1], im_shape[0])
    return val_inds

def show_kp(img, x, scale=1):
    plt.figure(figsize=(30*scale, 8*scale))
    plt.imshow(img)
    plt.scatter(x[:, 0], x[:, 1], s=10, marker='o', c='y')
    plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def draw_corr(im1, im2, x1, x2, linewidth):
    # im1 = img1_rgb
    # im2 = img2_rgb
    # x1 = x1_sample
    # x2 = x2_sample
    im_shape = im1.shape
    assert im1.shape == im2.shape, 'Shape mismatch between im1 and im2! @draw_corr()'
    x2_copy = x2.copy()
    x2_copy[:, 0] = x2_copy[:, 0] + im_shape[1]
    im12 = np.hstack((im1, im2))

    plt.figure(figsize=(30, 4))
    plt.imshow(im12)
    plt.plot(np.vstack((x1[:, 0], x2_copy[:, 0])), np.vstack((x1[:, 1], x2_copy[:, 1])), marker='o', linewidth=linewidth)
    plt.show()

def draw_corr_widths(im1, im2, x1, x2, linewidth, title='', rescale=True):
    # im1 = img1_rgb
    # im2 = img2_rgb
    # x1 = x1_sample
    # x2 = x2_sample
    im_shape = im1.shape
    assert im1.shape == im2.shape, 'Shape mismatch between im1 and im2! @draw_corr()'
    x2_copy = x2.copy()
    x2_copy[:, 0] = x2_copy[:, 0] + im_shape[1]
    im12 = np.hstack((im1, im2))

    plt.figure(figsize=(60, 8))
    plt.imshow(im12)
    for i in range(x1.shape[0]):
        if rescale:
            width = 5 if linewidth[i]<2 else 10
        else:
            width = linewidth[i]
        plt.plot(np.vstack((x1[i, 0], x2_copy[i, 0])), np.vstack((x1[i, 1], x2_copy[i, 1])), linewidth=width, marker='o', markersize=8)
    plt.title(title, {'fontsize':40})
    plt.show()

def reproj_and_scatter(Rt, X_rect, im_rgb, kitti_two_frame_loader, visualize=True, title_appendix=''):
    x1_homo = np.matmul(kitti_two_frame_loader.K, np.matmul(Rt, utils_misc.homo_np(X_rect.T).T)).T
    x1 = x1_homo[:, 0:2]/x1_homo[:, 2:3]
    if visualize:
        plt.figure(figsize=(30, 8))
        plt.imshow(im_rgb)
        val_inds = scatter_xy(x1, x1_homo[:, 2], kitti_two_frame_loader.im_shape, 'Reprojection to cam 2 with rectified X and camera_'+title_appendix, new_figure=False)
    else:
        val_inds = utils_misc.within(x1[:, 0], x1[:, 1], kitti_two_frame_loader.im_shape[1], kitti_two_frame_loader.im_shape[0])
    return val_inds
