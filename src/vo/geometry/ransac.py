import numpy as np
from .epipolar import eight_point_algorithm, sampson_distance

def estimate_essential_ransac(pts1, pts2, K, px_tol=1.0, threshold=None,
                               max_iters=1000, min_inliers=12, confidence=0.999):
    K_inv = np.linalg.inv(K)
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
    pts1_n = (K_inv @ pts1_h.T).T[:, :2]
    pts2_n = (K_inv @ pts2_h.T).T[:, :2]

    n = len(pts1_n)
    if n < 8:
        return None, None
    
    if threshold is None:
        f = (K[0, 0] + K[1, 1]) / 2.0
        threshold = (px_tol / f) ** 2

    best_inliers = None
    best_E = None
    rng = np.random.default_rng()

    iters = max_iters
    i = 0

    while i < iters:
        i += 1
        idx = rng.choice(n, size=8, replace=False) #без повторов

        try:
            E_candidate = eight_point_algorithm(pts1_n[idx], pts2_n[idx], essential=True)
        except np.linalg.LinAlgError:
            continue

        d = sampson_distance(E_candidate, pts1_n, pts2_n)
        inliers = d < threshold

        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_E = E_candidate

            w = best_inliers.sum() / n
            if w > 0:
                denom = 1.0 - w ** 8
                if 1e-12 < denom < 1.0 - 1e-12:
                    need = np.log(1.0 - confidence) / np.log(denom)
                    iters = min(max_iters, max(20, int(np.ceil(need))))

    if best_inliers is None or best_inliers.sum() < min_inliers:
        return None, None

    E_final = eight_point_algorithm(pts1_n[best_inliers], pts2_n[best_inliers], essential=True)

    return E_final, best_inliers