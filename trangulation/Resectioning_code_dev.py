import numpy as np

def resection(u, U, epsperc=0.95, maxerror=15, maxiter=500):
    print("\n\n******** Starting Resectioning Algorithm ********\n\n")

    # Setup termination criteria
    epsperc = 1 - epsperc

    sourceD = U.shape[0]  # sourceD = 3 for 3D points
    imageD = u.shape[0]  # imageD = 2 for 2D points
    nbrpoints = u.shape[1]

    # Translate image centroid to origin (and rescale coordinates)
    mm = np.mean(u, axis=1)
    ut = u - mm[:, np.newaxis]
    imscale = np.std(ut)
    ut /= imscale

    K = np.diag(np.append(np.ones(imageD) / imscale, 1))
    K[:imageD, -1] = -mm / imscale

    # Translate source points such that U_1 = [0, 0, ..., 1]^T
    mm = U[:, 0]
    Ut = U - mm[:, np.newaxis]
    tmp = Ut[:, 1:]
    ss = np.std(tmp)
    Ut /= ss

    T = np.diag(np.append(np.ones(sourceD) / ss, 1))
    T[:sourceD, -1] = -mm / ss

    # Upper & lower bounds
    thr = maxerror / imscale
    rect_yL, rect_yU = l2_reconstruct_bound(ut, Ut, thr)
    xL, xU = 0, 2 * thr

    hh = l2_reconstruct_loop(ut, Ut, rect_yL, rect_yU, xL, xU)

    # Stack Ut and ones for homogenous coordinates
    Ut_homog = np.vstack((Ut, np.ones((1, nbrpoints))))

    # Print the shapes to debug the error
    print(f"hh['H'] shape: {hh['H'].shape}")  # Expecting (3, 4)
    print(f"Ut shape: {Ut.shape}")  # Expecting (3, nbrpoints)
    print(f"np.vstack((Ut, np.ones((1, nbrpoints)))) shape: {Ut_homog.shape}")  # Expecting (4, nbrpoints)

    # Perform matrix multiplication
    tmp = hh["H"] @ Ut_homog  # hh["H"] should be a 3x4 matrix, Ut_homog is 4xN
    tmp = tmp[:imageD, :] / tmp[-1, :]  # Normalize by the last row
    tmp = tmp - ut

    # Update xU
    xU = np.max(np.abs(tmp)) * 2

    vopt = np.sum(hh["res"])
    H = hh["H"]

    rect_LB = hh["lowerbound"]
    rect = [hh]

    iter = 1
    while iter <= maxiter:
        vkindex = np.argmin(rect_LB)
        perc = (vopt - rect_LB[vkindex]) / vopt

        print(f"Iter: {iter} Residual: {vopt * imscale} Approximation gap: {perc * 100}% Regions: {len(rect)}")

        if perc < epsperc:
            break

        h = rect[vkindex]
        pp = np.argmax(rect_yU[:sourceD, vkindex] - rect_yL[:sourceD, vkindex])

        tmpyL = rect_yL[pp, vkindex]
        tmpyU = rect_yU[pp, vkindex]

        # Branching strategy
        bestsol = h["lambda"][pp]
        alfa = 0.2

        if (bestsol - tmpyL) / (tmpyU - tmpyL) < alfa:
            newborder = tmpyL + (tmpyU - tmpyL) * alfa
        elif (tmpyU - bestsol) / (tmpyU - tmpyL) < alfa:
            newborder = tmpyU - (tmpyU - tmpyL) * alfa
        else:
            newborder = bestsol

        curr_yL1 = rect_yL[:, vkindex]
        curr_yU1 = rect_yU[:, vkindex]
        curr_yL2 = curr_yL1.copy()
        curr_yU2 = curr_yU1.copy()

        curr_yU1[pp] = newborder
        curr_yL2[pp] = newborder

        rect_yL = np.hstack((rect_yL[:, :vkindex], curr_yL1[:, np.newaxis], curr_yL2[:, np.newaxis], rect_yL[:, vkindex + 1:]))
        rect_yU = np.hstack((rect_yU[:, :vkindex], curr_yU1[:, np.newaxis], curr_yU2[:, np.newaxis], rect_yU[:, vkindex + 1:]))

        h1 = l2_reconstruct_loop(ut, Ut, curr_yL1, curr_yU1, xL, xU)
        h2 = l2_reconstruct_loop(ut, Ut, curr_yL2, curr_yU2, xL, xU)

        vopt1 = np.sum(h1["res"])
        vopt2 = np.sum(h2["res"])

        rect = rect[:vkindex] + [h1, h2] + rect[vkindex + 1:]
        rect_LB = np.concatenate((
            rect_LB[:vkindex], 
            [np.squeeze(h1["lowerbound"]), np.squeeze(h2["lowerbound"])], 
            rect_LB[vkindex + 1:]
        ))

        if vopt1 < vopt:
            vopt = vopt1
            H = h1["H"]
        if vopt2 < vopt:
            vopt = vopt2
            H = h2["H"]

        # Remove useless regions
        removeindex = [ii for ii, r in enumerate(rect) if r["lowerbound"] > vopt]
        rect = [r for i, r in enumerate(rect) if i not in removeindex]
        rect_yL = np.delete(rect_yL, removeindex, axis=1)
        rect_yU = np.delete(rect_yU, removeindex, axis=1)
        rect_LB = np.delete(rect_LB, removeindex)

        iter += 1

    H = np.linalg.inv(K) @ H @ T
    H /= np.linalg.norm(H)

    print("******** Ending Resectioning Algorithm ********\n\n")
    return H

def l2_reconstruct_loop(ut, Ut, rect_yL, rect_yU, xL, xU):
    # Assuming H is a 3x4 matrix for homography
    H = np.random.rand(3, 4)  # This should be generated by your actual algorithm
    return {
        "H": H,
        "res": np.random.rand(ut.shape[1]),
        "lowerbound": np.random.rand(1),
        "lambda": np.random.rand(3)
    }

def l2_reconstruct_bound(ut, Ut, thr):
    maxU = 5
    minL = 0.2
    sourceD = Ut.shape[0]
    imageD = ut.shape[0]
    nbrpoints = ut.shape[1]

    rect_yL = minL * np.ones((nbrpoints - 1, 1))
    rect_yU = maxU * np.ones((nbrpoints - 1, 1))

    return rect_yL, rect_yU


# Example matrices
matrix_3D = np.array([
   [-151.2949, -11.0825, 305.5970],
   [11.0055, 0.6453, -22.7716],
   [10.8754, -0.9293, -22.6187],
   [10.9020, 0.3734, -22.7424],
   [10.9311, 0.6740, -22.8171],
   [10.8135, 0.6027, -22.8858],
   [10.7740, 0.5672, -22.9100],
   [10.6880, 0.0845, -22.7866],
   [10.9646, 2.3279, -23.3198],
   [10.6883, 0.2924, -22.8306],
   [10.7021, 0.3688, -22.8584],
   [10.7507, 0.8409, -22.9668],
   [10.7507, 0.8409, -22.9668],
   [10.8923, 2.1328, -23.2703],
   [10.8890, 2.2206, -23.3278],
   [10.8524, 2.0039, -23.2845],
   [10.6614, 0.8320, -22.9970],
   [11.9260, 6.3526, -25.6860],
   [10.6483, 1.0735, -23.1522],
   [10.5369, 0.1091, -22.9732]
])

matrix_2D = np.array([
   [25.6147, 338.8417],
   [44.0775, 311.1743],
   [46.1970, 403.3918],
   [49.1522, 327.1137],
   [49.7834, 309.6051],
   [58.5445, 314.1909],
   [61.5142, 316.2076],
   [62.7857, 344.3026],
   [62.9705, 215.4968],
   [64.1244, 332.3965],
   [64.1475, 327.8912],
   [64.6529, 300.3238],
   [64.6529, 300.3238],
   [65.5220, 223.4185],
   [65.9394, 218.6653],
   [66.6882, 226.3897],
   [68.3290, 307.0475],
   [75.3467, 175.2301],
   [77.8130, 291.0905],
   [87.3249, 342.0126]
])

# Call the resection function with matrices
H = resection(matrix_2D.T, matrix_3D.T)
print("Computed camera matrix H:")
print(H)