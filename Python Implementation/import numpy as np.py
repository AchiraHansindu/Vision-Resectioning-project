import numpy as np
from scipy.sparse import csr_matrix, vstack

# Define the matrices
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
    [65.6646, 226.4794],
    [67.4484, 221.9323],
    [68.2816, 233.6933],
    [70.6802, 301.0883],
    [73.8634, 12.8499],
    [75.7516, 288.0290],
    [76.7349, 343.3047]
])

# Parameters
epsilon = 0.95  # relative termination tolerance
delta = 15      # upper bound on the error
maxiter = 1000  # maximum iterations

# Define the resection function
def resection(u, U, epsperc=0.05, maxerror=15, maxiter=1000):
    print("\n\n******** Starting Resectioning Algorithm ********\n\n")
    epsperc = 1 - epsperc
    sourceD = U.shape[0]
    imageD = u.shape[0]
    nbrpoints = u.shape[1]
    epsdiff = 0.0

    # Translate image centroid to origin (and rescale coordinates)
    mm = np.mean(u, axis=1, keepdims=True)
    ut = u - mm
    imscale = np.std(ut)
    ut /= imscale

    K = np.diag([1 / imscale] * imageD + [1])
    K[:imageD, -1] = -mm.flatten() / imscale

    # Translate source points such that U_1=[0,...,0,1]^T (and rescale coordinates)
    mm = U[:, [0]]
    Ut = U - mm
    ss = np.std(Ut[:, 1:], axis=None)
    Ut /= ss

    T = np.diag([1 / ss] * sourceD + [1])
    T[:sourceD, -1] = -mm.flatten() / ss

    # Upper & lower bound...
    thr = maxerror / imscale
    rect_yL, rect_yU = l2_reconstruct_bound(ut, Ut, thr)
    xL = 0
    xU = 2 * thr

    # Initial estimate
    hh = l2_reconstruct_loop(ut, Ut, rect_yL, rect_yU, xL, xU)
    tmp = hh['H'] @ np.vstack([Ut, np.ones((1, nbrpoints))])
    tmp = tmp[:2, :] / tmp[2, :] - ut
    xU = np.max(np.abs(tmp) / np.min(rect_yL)) * 2

    vopt = np.sum(hh['res'])
    H = hh['H']
    rect_LB = [hh['lowerbound']]
    rect = [hh]

    # Branch and Bound-loop
    iter = 1
    while iter <= maxiter:
        vk, vkindex = min((val, idx) for (idx, val) in enumerate(rect_LB))
        vdiff = vopt - vk
        perc = vdiff / vopt

        print(f"Iter: {iter} Residual: {vopt * imscale} Approximation gap: {perc * 100}% Regions: {len(rect)}")

        if vdiff < epsdiff or perc < epsperc:
            break

        # Branch on vkindex
        h = rect[vkindex]
        pp = np.argmax(rect_yU[:sourceD, vkindex] - rect_yL[:sourceD, vkindex])

        tmpyL = rect_yL[pp, vkindex]
        tmpyU = rect_yU[pp, vkindex]
        
        # Branching strategy
        bestsol = h['lambda'][pp]
        alfa = 0.2
        if (bestsol - tmpyL) / (tmpyU - tmpyL) < alfa:
            newborder = tmpyL + (tmpyU - tmpyL) * alfa
        elif (tmpyU - bestsol) / (tmpyU - tmpyL) < alfa:
            newborder = tmpyU - (tmpyU - tmpyL) * alfa
        else:
            newborder = bestsol

        curr_yL1 = rect_yL[:, vkindex].copy()
        curr_yU1 = rect_yU[:, vkindex].copy()
        curr_yL2 = curr_yL1.copy()
        curr_yU2 = curr_yU1.copy()
        
        curr_yU1[pp] = newborder
        curr_yL2[pp] = newborder

        rect_yL = np.hstack([rect_yL[:, :vkindex], curr_yL1[:, np.newaxis], curr_yL2[:, np.newaxis], rect_yL[:, vkindex + 1:]])
        rect_yU = np.hstack([rect_yU[:, :vkindex], curr_yU1[:, np.newaxis], curr_yU2[:, np.newaxis], rect_yU[:, vkindex + 1:]])

        h1 = l2_reconstruct_loop(ut, Ut, curr_yL1, curr_yU1, xL, xU)
        h2 = l2_reconstruct_loop(ut, Ut, curr_yL2, curr_yU2, xL, xU)

        vopt1 = np.sum(h1['res'])
        vopt2 = np.sum(h2['res'])

        rect = rect[:vkindex] + [h1, h2] + rect[vkindex + 1:]
        rect_LB = rect_LB[:vkindex] + [h1['lowerbound'], h2['lowerbound']] + rect_LB[vkindex + 1:]

        if vopt1 < vopt:
            vopt = vopt1
            H = h1['H']
        if vopt2 < vopt:
            vopt = vopt2
            H = h2['H']

        # Remove useless regions
        removeindex = [i for i, r in enumerate(rect) if r['lowerbound'] > vopt]
        rect = [r for i, r in enumerate(rect) if i not in removeindex]
        rect_yL = np.delete(rect_yL, removeindex, axis=1)
        rect_yU = np.delete(rect_yU, removeindex, axis=1)
        rect_LB = [lb for i, lb in enumerate(rect_LB) if i not in removeindex]

        iter += 1

    # Final transformation
    H = np.linalg.inv(K) @ H @ T
    H /= np.linalg.norm(H)

    print("******** Ending Resectioning Algorithm ********\n\n")
    return H, iter

def l2_reconstruct_bound(ut, Ut, thr, rect_yL=None, rect_yU=None, updateindex=None):
    # Set initial bounds for depths if not provided
    maxU = 5  # maximum depth upper bound
    minL = 0.2  # minimum depth lower bound
    sourceD = Ut.shape[0]
    imageD = ut.shape[0]
    nbrpoints = ut.shape[1]

    if rect_yL is None or rect_yU is None:
        rect_yL = np.full((sourceD, 1), minL)
        rect_yU = np.full((sourceD, 1), maxU)
        updateindex = list(range(sourceD))

    # Placeholder for depth constraint logic
    # Actual implementation should compute and update rect_yL and rect_yU
    # based on depth constraints derived from ut and Ut

    # Example static update for demonstration purposes
    for i in updateindex:
        rect_yL[i] = minL  # updating with minimal logic
        rect_yU[i] = maxU  # updating with minimal logic

    return rect_yL, rect_yU

def l2_reconstruct_loop(ut, Ut, rect_yL, rect_yU, xL, xU):
    # Placeholder logic for reconstructing camera matrix H
    sourceD = Ut.shape[0]
    imageD = ut.shape[0]
    nbrpoints = ut.shape[1]

    # Create a homogeneous version of Ut for matrix operations
    Ut_homog = np.vstack([Ut, np.ones((1, nbrpoints))])

    # Initial random guess for H for demonstration purposes
    H = np.random.rand(imageD+1, sourceD+1)

    # Calculate the residuals and a dummy lower bound for the optimization region
    projected = H @ Ut_homog
    residuals = np.linalg.norm(ut - projected[:imageD, :] / projected[imageD, :], axis=0)
    lower_bound = np.sum(residuals)  # Simple sum of residuals as a lower bound

    # Dummy output structure, replace with actual optimization results
    hh = {
        'H': H,                      # Random H matrix
        'res': residuals,            # Computed residuals
        'lowerbound': lower_bound,   # Computed lower bound
        'lambda': np.random.rand(nbrpoints)  # Dummy lambda values
    }

    return hh

def calculate_bounds(Utmp, Ut, rect_yL, rect_yU, sourceD, cnt):
    # Assuming Utmp and Ut are correctly formatted
    yL = np.max([0.1, np.min(Ut[cnt]) - 1])  # Setting minimal depth with a buffer
    yU = np.max([5, np.max(Ut[cnt]) + 1])    # Setting maximal depth with a buffer
    return yL, yU

from scipy.sparse import lil_matrix

def setup_linear_constraints(Utmp, yL, yU, xL, xU, index, indexa):
    # Assuming an optimization context where bounds on variables are required
    n_constraints = 3
    Atmp = lil_matrix((n_constraints, 10))  # Adjust the size based on your variables
    ctmp = np.zeros(n_constraints)
    
    # Constraint 1: x[index] should be greater than or equal to yL
    Atmp[0, index] = -1
    ctmp[0] = -yL
    
    # Constraint 2: x[index] should be less than or equal to yU
    Atmp[1, index] = 1
    ctmp[1] = yU
    
    # Constraint 3: example linear constraint
    Atmp[2, index] = 2
    Atmp[2, indexa] = -3
    ctmp[2] = xU
    
    return Atmp, ctmp, n_constraints

def process_solution(feasible, info, y, Ut, ut, Hvars, index_z, imageD, sourceD, nbrpoints):
    if not feasible or info.get('dinf', False):
        print("No feasible solution found.")
        return None
    
    H = np.eye(imageD + 1, sourceD + 1)  # Reset H to identity matrix with adjustment
    H[:-1, :] = y.reshape((imageD, sourceD + 1))
    
    # Compute projections and residuals
    proj = H @ np.vstack([Ut, np.ones((1, nbrpoints))])
    residuals = np.linalg.norm(ut - proj[:-1, :] / proj[-1, :], axis=0)
    lowerbound = np.sum(residuals)
    
    return {
        'H': H,
        'residuals': residuals,
        'lower_bound': lowerbound,
        'depths': y[index_z]  # Assuming y contains depths or similar at index_z
    }

# Function to execute the resection process and print the results
def execute_resection():
    u = matrix_2D.T  # Transpose to fit expected dimensions if needed
    U = matrix_3D.T  # Transpose to fit expected dimensions if needed
    
    # Call the resection function
    H, iterations = resection(u, U, epsilon, delta, maxiter)
    
    # Print the estimated camera matrix and other outputs
    print("Estimated Camera Matrix (H):\n", H)
    print("Number of Iterations:", iterations)

# Main execution
if __name__ == "__main__":
    execute_resection()
