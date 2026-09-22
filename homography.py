import numpy as np

def solveAXEqualsZero(A):
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    h = h / h[-1]
    return h

def calcBestHomography(pts1Cart, pts2Cart):
    n_points = pts1Cart.shape[1]
    pts1Hom = np.vstack((pts1Cart, np.ones((1, n_points))))
    pts2Hom = np.vstack((pts2Cart, np.ones((1, n_points))))
    H = np.identity(3)
    A = []
    for i in range(n_points):
        x, y, w = pts1Hom[:, i]
        xp, yp, wp = pts2Hom[:, i]
        A.append([0, 0, 0, -wp * x, -wp * y, -wp * w, yp * x, yp * y, yp * w])
        A.append([wp * x, wp * y, wp * w, 0, 0, 0, -xp * x, -xp * y, -xp * w])
    A = np.array(A)
    h = solveAXEqualsZero(A)
    H = h.reshape((3, 3))
    return H

def estimatePlanePose(XImCart, XCart, K):
    """
    Estimate extrinsic matrix T (4x4) mapping plane coordinates XCart -> camera coordinates,
    given image points XImCart and intrinsics K.
    """
    n_points = XImCart.shape[1]
    XImHom = np.vstack((XImCart, np.ones((1, n_points))))
    XCamHom = np.linalg.inv(K) @ XImHom
    XCamCart = XCamHom[:2, :] / XCamHom[2, :]
    H = calcBestHomography(XCart[:2, :], XCamCart)
    B = H.copy()
    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]
    lambda_scale = 1.0 / ((np.linalg.norm(b1) + np.linalg.norm(b2)) / 2.0)
    r1 = lambda_scale * b1
    r2 = lambda_scale * b2
    t = lambda_scale * b3
    r3 = np.cross(r1, r2)
    R_approx = np.column_stack((r1, r2, r3))
    U, S, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    s1 = np.linalg.norm(b1)
    s2 = np.linalg.norm(b2)
    if s1 + s2 == 0:
        k = 1.0
    else:
        k = (s1 + s2) / 2.0
    t = lambda_scale * b3
    if t[2] < 0:
        R[:, 0:2] = -R[:, 0:2]
        t = -t
    T = np.vstack((np.column_stack((R, t)), [0.0, 0.0, 0.0, 1.0]))
    return T

def projectPoints(X, T, K):
    """
    Projects 3D points X (3xN) into the image using extrinsics T and intrinsics K
    """
    X_h = np.vstack((X, np.ones((1, X.shape[1]))))
    x_proj = K @ T[:3, :] @ X_h
    x_proj /= x_proj[2, :]
    return x_proj[:2, :]
