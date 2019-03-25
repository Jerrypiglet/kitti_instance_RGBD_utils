import cv2
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
import dsac_tools.utils_vis as utils_vis
import dsac_tools.utils_misc as utils_misc
import dsac_tools.utils_F as utils_F
import dsac_tools.utils_geo as utils_geo

def PIL_to_gray(im_PIL):
    img1_rgb = np.array(im_PIL)
    img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2GRAY)
    return img1

def SIFT_det(img, img_rgb, visualize=False, nfeatures=2000):
    # Initiate SIFT detector
    # pip install opencv-python==3.4.2.16, opencv-contrib-python==3.4.2.16
    # https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures, contrastThreshold=1e-5)

    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img,None)
    print("# kps: {}, descriptors: {}".format(len(kp), des.shape))
    x_all = np.array([p.pt for p in kp])

    if visualize:
        plt.figure(figsize=(30, 4))
        plt.imshow(img_rgb)
        plt.scatter(x_all[:, 0], x_all[:, 1], s=10, marker='o', c='y')
        plt.show()

    return x_all, kp, des

def KNN_match(des1, des2, x1_all, x2_all, kp1, kp2, img1_rgb, img2_rgb, visualize=False, if_BF=False):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # des2 = np.vstack((des2, np.zeros((1, 128), dtype=des2.dtype)))
    # print(des1.shape, des2.shape)
    if if_BF: # Use brute force matching
        bf = cv2.BFMatcher(normType=cv2.NORM_L2)
        matches = bf.knnMatch(des1,des2, k=2)
        # # Apply ratio test
        # good = []
        # for m,n in matches:
        #     if m.distance < 0.7*n.distance:
        #         good.append(m)
    else:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2) # another option is https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py#L309
    # print(matches)
    # store all the good matches as per Lowe's ratio test.
    good = []
    all_m = []
    for m,n in matches:
        all_m.append(m)
        if m.distance < 0.7*n.distance:
            good.append(m)

    x1 = x1_all[[mat.queryIdx for mat in good], :]
    x2 = x2_all[[mat.trainIdx for mat in good], :]
    assert x1.shape == x2.shape

    print('# good points: ', len(good))

    if visualize:
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = None, # draw only inliers
                           flags = 2)
        img3 = cv2.drawMatches(PIL_to_gray(img1_rgb), kp1, PIL_to_gray(img2_rgb), kp2, good, None, **draw_params)

        plt.figure(figsize=(60, 8))
        plt.imshow(img3, 'gray')
        plt.show()

        plt.figure(figsize=(30, 8))
        plt.imshow(img1_rgb)
        plt.scatter(x1[:, 0], x1[:, 1], s=10, marker='o', c='y')
        plt.title('Good points, #%d'%x1.shape[0])
        plt.show()

        plt.figure(figsize=(30, 8)) 
        plt.imshow(img2_rgb)
        plt.scatter(x2[:, 0], x2[:, 1], s=10, marker='o', c='y')
        plt.title('Good points, #%d'%x1.shape[0])
        plt.show()

    good_ij = [[mat.queryIdx for mat in good], [mat.trainIdx for mat in good]]
    all_ij = [[mat.queryIdx for mat in all_m], [mat.trainIdx for mat in all_m]]

    return x1, x2, np.asarray(all_ij).T.copy(), np.asarray(good_ij).T.copy()

def show_epipolar_opencv(x1, x2, img1_rgb, img2_rgb, F_gt):
    lines2 = cv2.computeCorrespondEpilines(x1.reshape(-1,1,2).astype(int), 1,F_gt)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = utils_vis.drawlines(np.array(img2_rgb).copy(), np.array(img1_rgb).copy(), lines2, x2.astype(int), x1.astype(int))

    plt.figure(figsize=(30, 8))
    # plt.subplot(121)
    plt.imshow(img4)
    # plt.subplot(122),
    plt.figure(figsize=(30, 8))
    plt.imshow(img3)
    plt.show()
    return

def show_epipolar_rui(x1, x2, img1_rgb, img2_rgb, F_gt, im_shape):
    N_points = x1.shape[0]
    x1_homo = utils_misc.homo_np(x1)
    x2_homo = utils_misc.homo_np(x2)
    right_P = np.matmul(F_gt, x1_homo.T)
    right_epipolar_x = np.tile(np.array([[0], [1]]), N_points) * im_shape[1]
    # Using the eqn of line: ax+by+c=0; y = (-c-ax)/b, http://ai.stanford.edu/~mitul/cs223b/draw_epipolar.m
    right_epipolar_y = (-right_P[2:3, :] - right_P[0:1, :] * right_epipolar_x) / right_P[1:2, :]

    colors = np.random.rand(x2.shape[0])
    plt.figure(figsize=(30, 8))
    plt.subplot(121)
    plt.imshow(img1_rgb)
    plt.scatter(x1[:, 0], x1[:, 1], s=50, c=colors, edgecolors='w')
    plt.subplot(122)
    # plt.figure(figsize=(30, 8))
    plt.imshow(img2_rgb)
    plt.plot(right_epipolar_x, right_epipolar_y)
    plt.scatter(x2[:, 0], x2[:, 1], s=50, c=colors, edgecolors='w')
    plt.xlim(0, im_shape[1]-1)
    plt.ylim(im_shape[0]-1, 0)
    plt.show()

def show_epipolar_normalized(x1, x2, img1_rgb, img2_rgb, F_gt, im_shape):
    N_points = x1.shape[0]
    x1_homo = utils_misc.homo_np(x1)
    x2_homo = utils_misc.homo_np(x2)
    right_P = np.matmul(F_gt, x1_homo.T)
    right_epipolar_x = np.tile(np.array([[-1.], [1.]]), N_points) * im_shape[1]
    # Using the eqn of line: ax+by+c=0; y = (-c-ax)/b, http://ai.stanford.edu/~mitul/cs223b/draw_epipolar.m
    right_epipolar_y = (-right_P[2:3, :] - right_P[0:1, :] * right_epipolar_x) / right_P[1:2, :]

    colors = np.random.rand(x2.shape[0])
    plt.figure(figsize=(30, 8))
    plt.subplot(121)
#     plt.imshow(img1_rgb)
#     plt.scatter(x1[:, 0]*f+W/2., x1[:, 1]*f+H/2., s=50, c=colors, edgecolors='w')
    plt.scatter(x1[:, 0], x1[:, 1], s=50, c=colors, edgecolors='w')
    plt.xlim(-im_shape[1], im_shape[1])
    plt.ylim(im_shape[0], -im_shape[0])
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.subplot(122)
#     plt.imshow(img2_rgb)
    plt.plot(right_epipolar_x, right_epipolar_y)
    plt.scatter(x2[:, 0], x2[:, 1], s=50, c=colors, edgecolors='w')
#     plt.axis('equal')
    plt.xlim(-im_shape[1], im_shape[1])
    plt.ylim(im_shape[0], -im_shape[0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def sample_and_check(x1, x2, img1_rgb, img2_rgb, img1_rgb_np, img2_rgb_np, F_gt, kitti_two_frame_loader, visualize=False):
    import random
    # random.seed(10)
    N_points = 50
    random_idx = random.sample(range(x1.shape[0]), N_points)
    # random_idx = mask_index
    x1_sample = x1[random_idx, :]
    x2_sample = x2[random_idx, :]

    if visualize:
        print('>>>>>>>>>>>>>>>> Sample points, and check epipolar and Sampson distances. ---------------')

        ## Draw epipolar lines: by OpenCV
        show_epipolar_opencv(x1_sample, x2_sample, img1_rgb, img2_rgb, F_gt)

        ## Draw epipolar lines: by Rui
        show_epipolar_rui(x1_sample, x2_sample, img1_rgb, img2_rgb, F_gt, kitti_two_frame_loader.im_shape)
            
        ## Show corres.
        utils_vis.draw_corr(img1_rgb_np, img2_rgb_np, x1_sample, x2_sample, 2)

        ## Show Sampson distances
        sampson_dist_gtF = utils_F._sampson_dist(torch.from_numpy(F_gt), torch.from_numpy(x1_sample), torch.from_numpy(x2_sample), False)
        print(sampson_dist_gtF.numpy())
        sampson_dist_gtF_plot = np.log(sampson_dist_gtF.numpy()+1)+1
        utils_vis.draw_corr_widths(img1_rgb_np, img2_rgb_np, x1_sample, x2_sample, sampson_dist_gtF_plot, 'Sampson distance w.r.t. ground truth F (the thicker the worse corres.)', False)

        print('<<<<<<<<<<<<<<<< DONE. Sample points, and check epipolar and Sampson distances. ---------------')

    return random_idx, x1_sample, x2_sample

def recover_camera_opencv(E_gt, K, x1, x2, delta_Rtij_inv, five_point=False, threshold=0.1):
    # Compare with OpenCV with refs from:
    ## https://github.com/vcg-uvic/learned-correspondence-release/blob/16bef8a0293c042c0bd42f067d7597b8e84ef51a/tests.py#L232
    ## https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    ## http://answers.opencv.org/question/90070/findessentialmat-or-decomposeessentialmat-do-not-work-correctly/

    print('>>>>>>>>>>>>>>>> Running OpenCV camera pose estimation... ---------------')

    # Mostly following: # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv

    E_5point, mask = cv2.findEssentialMat(x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), method=cv2.RANSAC, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 

    # x1_norm = cv2.undistortPoints(np.expand_dims(x1, axis=1), cameraMatrix=K, distCoeffs=None) 
    # x2_norm = cv2.undistortPoints(np.expand_dims(x2, axis=1), cameraMatrix=K, distCoeffs=None)
    # E_5point, mask = cv2.findEssentialMat(x1_norm, x2_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 

    F_8point, _ = cv2.findFundamentalMat(x1, x2, method=cv2.RANSAC) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
    E_8point = K.T @ F_8point @ K
    U,S,V = np.linalg.svd(E_8point)
    E_8point = U @ np.diag([1., 1., 0.]) @ V

    method_name = '5 point' if five_point else '8 point'
    E_recover = E_5point if five_point else E_8point
    # E_recover = E_5point
    # if five_point:
    points, R, t, mask = cv2.recoverPose(E_recover, x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]))
    # print(R, t)
    # else:
        # points, R, t, mask = cv2.recoverPose(E_recover, x1, x2)
    print('# %d/%d inliers from OpenCV.'%(np.sum(mask==255), mask.shape[0]))

    # R_cam, t_cam = utils_geo.invert_Rt(R, t)

    error_R = utils_geo.rot12_to_angle_error(R, delta_Rtij_inv[:3, :3])
    error_t = utils_geo.vector_angle(t, delta_Rtij_inv[:3, 3:4])
    print('Recovered by OpenCV %s (camera): The rotation error (degree) %.4f, and translation error (degree) %.4f'%(method_name, error_R, error_t))
    print(np.hstack((R, t)))

    # M_r = np.hstack((R, t))
    # M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    # P_l = np.dot(K,  M_l)
    # P_r = np.dot(K,  M_r)
    # point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1))
    # point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    # point_3d = point_4d[:3, :].T
    # scipy.io.savemat('test.mat', {'X': point_3d})

    print('<<<<<<<<<<<<<<<< DONE Running OpenCV camera pose estimation. ---------------')

    return np.hstack((R, t)), (error_R, error_t)
        
# def recover_camera(E_gt, K, x1, x2, delta_Rtij_inv, five_point=False, threshold=0.1):
#     # Compare with OpenCV with refs from:
#     ## https://github.com/vcg-uvic/learned-correspondence-release/blob/16bef8a0293c042c0bd42f067d7597b8e84ef51a/tests.py#L232
#     ## https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
#     ## http://answers.opencv.org/question/90070/findessentialmat-or-decomposeessentialmat-do-not-work-correctly/

#     # E, mask = cv2.findEssentialMat(x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), method=cv2.RANSAC, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 

#     # x1_norm = cv2.undistortPoints(np.expand_dims(x1, axis=1), cameraMatrix=K, distCoeffs=None) 
#     # x2_norm = cv2.undistortPoints(np.expand_dims(x2, axis=1), cameraMatrix=K, distCoeffs=None)
#     # E_5point, mask = cv2.findEssentialMat(x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), method=cv2.RANSAC, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
#     E, mask = cv2.findEssentialMat(x1, x2, method=cv2.RANSAC, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 

#     points, R, t, mask = cv2.recoverPose(E, x1, x2)
#     print('# %d/%d inliers from OpenCV.'%(np.sum(mask==255), mask.shape[0])) 

#     R_cam, t_cam = utils_geo.invert_Rt(R, t)

#     error_R = utils_geo.rot12_to_angle_error(R, delta_Rtij_inv[:, :3])
#     error_t = utils_geo.vector_angle(t, delta_Rtij_inv[:, 3:4])
#     print('Recovered by OpenCV: The rotation error (degree) %.4f, and translation error (degree) %.4f'%(error_R, error_t))
#     print(np.hstack((R, t)))

#     return np.hstack((R, t)), (error_R, error_t)

# def recover_camera_0(E_gt, K, x1, x2, delta_Rtij_inv, five_point=False, threshold=0.1):
#     # Compare with OpenCV with refs from:
#     ## https://github.com/vcg-uvic/learned-correspondence-release/blob/16bef8a0293c042c0bd42f067d7597b8e84ef51a/tests.py#L232
#     ## https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
#     ## http://answers.opencv.org/question/90070/findessentialmat-or-decomposeessentialmat-do-not-work-correctly/
#     E, mask = cv2.findEssentialMat(x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), method=cv2.RANSAC, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
#     points, R, t, mask = cv2.recoverPose(E, x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]))
#     print(t)
#     print('# %d/%d inliers from OpenCV.'%(np.sum(mask==255), mask.shape[0])) 

#     # R_cam, t_cam = utils_geo.invert_Rt(R, t)
#     # print(t_cam)

#     error_R = utils_geo.rot12_to_angle_error(R, delta_Rtij_inv[:, :3])
#     error_t = utils_geo.vector_angle(t, delta_Rtij_inv[:, 3:4])
#     print('Recovered by OpenCV: The rotation error (degree) %.4f, and translation error (degree) %.4f'%(error_R, error_t))
#     print(np.hstack((R, t)))
#     return np.hstack((R, t)), (error_R, error_t)

# def recover_camera_1(E_gt, K, x1, x2, delta_Rtij_inv, five_point=False, threshold=0.1):
#     # Compare with OpenCV with refs from:
#     ## https://github.com/vcg-uvic/learned-correspondence-release/blob/16bef8a0293c042c0bd42f067d7597b8e84ef51a/tests.py#L232
#     ## https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
#     ## http://answers.opencv.org/question/90070/findessentialmat-or-decomposeessentialmat-do-not-work-correctly/

#     # x1_norm = cv2.undistortPoints(np.expand_dims(x1, axis=1), cameraMatrix=K, distCoeffs=None) 
#     # x2_norm = cv2.undistortPoints(np.expand_dims(x2, axis=1), cameraMatrix=K, distCoeffs=None)

#     x1 = utils_misc.de_homo_np((np.linalg.inv(K) @ (utils_misc.homo_np(x1).T)).T)
#     print(x1, x1.shape)
#     x2 = utils_misc.de_homo_np((np.linalg.inv(K) @ (utils_misc.homo_np(x2).T)).T)
#     E_8point, _ = cv2.findFundamentalMat(x1, x2, method=cv2.RANSAC) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
#     # print(F_8point)
#     # E_8point = K.T @ F_8point @ K
#     U,S,V = np.linalg.svd(E_8point)
#     print(S)
#     E = U @ np.diag([1., 1., 0.]) @ V

#     # E, mask = cv2.findEssentialMat(x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), method=cv2.RANSAC, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
#     points, R, t, mask = cv2.recoverPose(E, x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]))
#     # points, R, t, mask = cv2.recoverPose(E, x1, x2, focal=1., pp=(0., 0.))

#     print(t)
#     print('# %d/%d inliers from OpenCV.'%(np.sum(mask==255), mask.shape[0])) 

#     # R_cam, t_cam = utils_geo.invert_Rt(R, t)
#     # print(t_cam)

#     error_R = utils_geo.rot12_to_angle_error(R, delta_Rtij_inv[:, :3])
#     error_t = utils_geo.vector_angle(t, delta_Rtij_inv[:, 3:4])
#     print('Recovered by OpenCV: The rotation error (degree) %.4f, and translation error (degree) %.4f'%(error_R, error_t))
#     print(np.hstack((R, t)))
#     return np.hstack((R, t)), (error_R, error_t)

def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
