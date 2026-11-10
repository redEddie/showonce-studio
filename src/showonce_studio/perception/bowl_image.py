"""Known upright bowl silhouette + independently measured support plane.

The bowl surface may have no valid depth at all. Its known-size frustum is
matched to the RGB segmentation outline; surrounding depth supplies the table.
This is conditional geometry, not a general 6D pose estimator. Silhouette
agreement alone cannot certify every lift/tilt from a single camera.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares


class BowlRejected(ValueError):
    def __init__(self, reason, checks=None):
        super().__init__(reason)
        self.reason, self.checks = reason, checks or {}


def project(points, K, T_base_cam):
    camera = (points-T_base_cam[:3, 3]) @ T_base_cam[:3, :3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= .02):
        raise BowlRejected('model_behind_camera')
    pixels = camera @ K.T
    return pixels[:, :2]/pixels[:, 2, None]


def basis(normal):
    u = np.array([1., 0., 0.])-normal[0]*normal
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def silhouette(centre, normal, size, K, T_base_cam):
    u, v = basis(normal)
    angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
    ring = np.cos(angles)[:, None]*u+np.sin(angles)[:, None]*v
    points = np.concatenate([centre+size['bottom_radius']*ring,
                             centre+size['height']*normal+size['rim_radius']*ring])
    pixels = project(points, K, T_base_cam)
    return cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2).astype(float)


def resample(polygon, count=96):
    following = np.roll(polygon, -1, axis=0)
    length = np.linalg.norm(following-polygon, axis=1)
    cumulative = np.r_[0., np.cumsum(length)]
    if cumulative[-1] <= 1e-6:
        raise BowlRejected('degenerate_outline')
    distances = np.linspace(0., cumulative[-1], count, endpoint=False)
    segment = np.minimum(np.searchsorted(cumulative, distances, side='right')-1, len(length)-1)
    fraction = (distances-cumulative[segment])/np.maximum(length[segment], 1e-9)
    return polygon[segment]+fraction[:, None]*(following[segment]-polygon[segment])


def boundary_distances(points, polygon):
    delta = np.roll(polygon, -1, axis=0)-polygon
    relative = points[:, None]-polygon
    fraction = np.clip(np.sum(relative*delta, axis=-1)/np.maximum(np.sum(delta*delta, axis=-1), 1e-12), 0., 1.)
    residual = relative-fraction[:, :, None]*delta
    return np.sqrt(np.maximum(np.min(np.sum(residual*residual, axis=-1), axis=1), 0.))


def support_plane(mask, depth, K, T_base_cam, hint_z, exclusions=None):
    outside_distance = cv2.distanceTransform(1-mask, cv2.DIST_L2, 5)
    ring = (outside_distance > 10) & (outside_distance <= 55)
    if exclusions is not None:
        ring &= ~np.asarray(exclusions, bool)
    y, x = np.nonzero(ring & np.isfinite(depth) & (depth > .1) & (depth < 2.))
    if len(y) < 180:
        raise BowlRejected('table_depth_insufficient', dict(table_valid_points=int(len(y))))
    indices = np.linspace(0, len(y)-1, min(len(y), 1400)).astype(int)
    x, y = x[indices], y[indices]
    z = depth[y, x]
    rays = np.c_[x, y, np.ones(len(x))] @ np.linalg.inv(K).T
    points = (rays*z[:, None]) @ T_base_cam[:3, :3].T+T_base_cam[:3, 3]
    near = np.abs(points[:, 2]-hint_z) < .025
    candidates = points[near]
    if len(candidates) < 180 or np.mean(near) < .4:
        raise BowlRejected('table_plane_not_supported')
    rng = np.random.default_rng(100)
    triples = candidates[rng.integers(0, len(candidates), (48, 3))]
    normals = np.cross(triples[:, 1]-triples[:, 0], triples[:, 2]-triples[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(norms[:, None], 1e-12)
    normals *= np.where(normals[:, 2:3] < 0, -1., 1.)
    possible = (norms > 1e-9) & (normals[:, 2] > np.cos(np.deg2rad(5.)))
    if not np.any(possible):
        raise BowlRejected('table_plane_not_horizontal')
    normals, triples = normals[possible], triples[possible]
    offsets = np.sum(normals*triples[:, 0], axis=1)
    distances = np.abs(points @ normals.T-offsets)
    winner = np.argmax(np.sum(distances < .003, axis=0))
    inliers = distances[:, winner] < .003
    fraction = float(np.mean(inliers))
    if fraction < .30:
        raise BowlRejected('table_plane_not_supported', dict(table_inlier_fraction=fraction,
                                                            table_inlier_points=int(np.sum(inliers))))
    selected = points[inliers]
    centre = np.mean(selected, axis=0)
    normal = np.linalg.svd(selected-centre, full_matrices=False)[2][-1]
    if normal[2] < 0:
        normal = -normal
    offset = float(normal @ centre)
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1., 1.))))
    origin_height = offset/normal[2]
    if tilt > 5. or abs(origin_height-hint_z) > .015:
        raise BowlRejected('table_plane_inconsistent_with_calibration')
    residual_p95_mm = float(np.percentile(np.abs(selected @ normal-offset), 95)*1000)
    span = [int(np.ptp(x[inliers])), int(np.ptp(y[inliers]))]
    sparse = fraction < .5
    # A cluttered ring can still contain a broad, accurate tabletop patch.
    # Its support must agree more tightly with calibration than a dense fit.
    if sparse and (len(selected) < 300 or min(span) < 80 or tilt > 3.
                   or abs(origin_height-hint_z) > .005 or residual_p95_mm > 3.):
        raise BowlRejected('table_plane_not_supported', dict(table_inlier_fraction=fraction,
                          table_inlier_points=int(len(selected)), table_inlier_pixel_span=span,
                          table_tilt_deg=tilt, table_origin_height_m=float(origin_height),
                          table_residual_p95_mm=residual_p95_mm))
    checks = dict(table_points=int(len(points)), table_inlier_fraction=fraction,
                  table_inlier_points=int(len(selected)), table_inlier_pixel_span=span,
                  table_plane_sparse_validated=sparse,
                  table_tilt_deg=tilt, table_origin_height_m=float(origin_height),
                  table_residual_p95_mm=residual_p95_mm)
    return normal, offset, checks


def estimate_bowl(definition, mask, depth, K, T_base_cam, *, table_hint_z,
                  rgb=None, exclusions=None, initial_centre=None, support=None):
    mask = np.asarray(mask, np.uint8)
    if mask.ndim != 2 or mask.shape != depth.shape or not np.isfinite(table_hint_z):
        raise BowlRejected('invalid_input')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise BowlRejected('empty_mask')
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    area = cv2.contourArea(contour)
    if area < 300 or area/sum(cv2.contourArea(c) for c in contours) < .95:
        raise BowlRejected('mask_fragmented_or_small')
    if (contour.min(0) < 3).any() or (contour.max(0) >= np.array(mask.shape[::-1])-3).any():
        raise BowlRejected('bowl_clipped_by_image')
    hull = cv2.convexHull(contour.astype(np.float32)).reshape(-1, 2)
    convexity = area/max(cv2.contourArea(hull), 1.)
    if convexity < .93:
        raise BowlRejected('bowl_outline_occluded_or_deformed', dict(mask_convexity=float(convexity)))
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, [contour], -1, 1, cv2.FILLED)
    if support is None:
        normal, offset, checks = support_plane(solid, depth, K, T_base_cam, table_hint_z, exclusions)
        checks['table_plane_fixed_for_session'] = False
    else:
        normal = np.asarray(support['normal'], float)
        offset = float(support['offset'])
        if (normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset)
                or abs(np.linalg.norm(normal)-1) > 1e-4 or normal[2] < np.cos(np.deg2rad(5.))
                or abs(offset/normal[2]-table_hint_z) > .015):
            raise BowlRejected('cached_table_plane_invalid')
        checks = dict(support['checks'], table_plane_fixed_for_session=True,
                      table_plane_source_time_s=support['source_time_s'])
    checks.update(method='image_silhouette_table_plane', upright_on_table_assumed=True,
                  pose_scope='conditional_upright_on_table_not_full_6D',
                  mask_convexity=float(convexity), mask_depth_valid_fraction=float(np.mean(depth[mask > 0] > 0)))
    if rgb is not None:
        edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), 40, 100)
        distances = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
        edge_support = float(np.mean(distances[contour[:, 1], contour[:, 0]] <= 3.))
        checks['outline_edge_support'] = edge_support
        if edge_support < .65:
            raise BowlRejected('RGB_outline_not_supported', checks)
    size = definition['size']
    if not (0 < size['bottom_radius'] < size['rim_radius'] < .3 and 0 < size['height'] < .3):
        raise BowlRejected('invalid_bowl_dimensions')
    u, v = basis(normal)
    if initial_centre is None:
        uv = np.mean(contour, axis=0)
        ray = T_base_cam[:3, :3] @ np.linalg.solve(K, np.r_[uv, 1.])
        if abs(normal @ ray) < 1e-6:
            raise BowlRejected('view_parallel_to_table')
        distance = (offset+.55*size['height']-normal @ T_base_cam[:3, 3])/(normal @ ray)
        origin = T_base_cam[:3, 3]+distance*ray-.55*size['height']*normal
    else:
        origin = np.asarray(initial_centre, float)
        origin = origin+(offset-normal @ origin)*normal
    observed = resample(contour.astype(float), 64)
    observed_polygon = cv2.approxPolyDP(contour, .35, True).reshape(-1, 2).astype(float)
    def outline(parameters):
        centre = origin+parameters[0]*u+parameters[1]*v
        return silhouette(centre, normal, size, K, T_base_cam)
    def residual(parameters):
        predicted = outline(parameters)
        return np.r_[boundary_distances(observed, predicted),
                     boundary_distances(resample(predicted, 64), observed_polygon)]
    def jacobian(parameters):
        step = .0001
        return np.column_stack([(residual(parameters+np.eye(2)[i]*step)-residual(parameters-np.eye(2)[i]*step))/(2*step) for i in range(2)])
    parameters = None
    if initial_centre is not None:
        # The mask is tracked from the current RGB frame. Reuse the last pose
        # only when its current-frame silhouette still matches tightly.
        prior_outline = outline(np.zeros(2))
        prior_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(prior_mask, np.rint(prior_outline).astype(np.int32), 1)
        prior_iou = np.count_nonzero((solid > 0) & (prior_mask > 0))/max(
            np.count_nonzero((solid > 0) | (prior_mask > 0)), 1)
        prior_error = float(np.percentile(residual(np.zeros(2)), 95))
        if prior_iou >= .91 and prior_error <= 2.75:
            parameters = np.zeros(2)
            checks['warm_pose_reused_from_current_rgb'] = True
    if parameters is None:
        result = least_squares(residual, np.zeros(2), jac=jacobian, bounds=(-.12, .12),
                               loss='soft_l1', f_scale=1., max_nfev=25, ftol=1e-4, xtol=1e-5, gtol=1e-3)
        parameters = result.x
        checks['warm_pose_reused_from_current_rgb'] = False
    centre = origin+parameters[0]*u+parameters[1]*v
    predicted = outline(parameters)
    model_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(model_mask, np.rint(predicted).astype(np.int32), 1)
    intersection = np.count_nonzero((solid > 0) & (model_mask > 0))
    union = np.count_nonzero((solid > 0) | (model_mask > 0))
    iou = intersection/max(union, 1)
    error = float(np.percentile(residual(parameters), 95))
    checks.update(silhouette_iou=float(iou), outline_p95_px=error,
                  table_normal_base=normal.tolist(), table_plane_offset_m=offset)
    if iou < .88 or error > 3.:
        raise BowlRejected('bowl_inconsistent_with_upright_table_model', checks)
    # Sparse bowl returns are optional, but clear evidence above the whole model
    # must not be explained away by the support assumption (e.g. an entering hand).
    y, x = np.nonzero((mask > 0) & np.isfinite(depth) & (depth > 0))
    if len(y) >= 30:
        ray = np.c_[x, y, np.ones(len(x))] @ np.linalg.inv(K).T
        points = (ray*depth[y, x, None]) @ T_base_cam[:3, :3].T+T_base_cam[:3, 3]
        contradiction = float(np.mean(points @ normal-offset > size['height']+.015))
        checks['depth_above_model_fraction'] = contradiction
        if contradiction > .25:
            raise BowlRejected('depth_contradicts_supported_bowl', checks)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([u, v, normal])
    transform[:3, 3] = centre
    return transform, float(centre[2]), checks


def track_bowl_mask(previous, current, mask):
    """Measured image motion; failure never fabricates a fresh stationary pose."""
    before = cv2.cvtColor(previous['bgr'], cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(current['bgr'], cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(before, 80, .005, 4, mask=np.asarray(mask, np.uint8)*255, blockSize=5)
    if points is None or len(points) < 8:
        raise BowlRejected('RGB_tracking_features_insufficient')
    forward, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None, winSize=(21, 21), maxLevel=3)
    if forward is None:
        raise BowlRejected('RGB_tracking_lost')
    backward, reverse, _ = cv2.calcOpticalFlowPyrLK(after, before, forward, None, winSize=(21, 21), maxLevel=3)
    if backward is None:
        raise BowlRejected('RGB_tracking_lost')
    good = status.ravel().astype(bool) & reverse.ravel().astype(bool)
    good &= np.linalg.norm(backward-points, axis=(1, 2)) < 1.2
    if good.sum() < 8 or np.mean(good) < .55:
        raise BowlRejected('RGB_tracking_inconsistent')
    affine, inliers = cv2.estimateAffinePartial2D(points[good], forward[good], method=cv2.RANSAC, ransacReprojThreshold=2.)
    if affine is None or inliers is None or np.mean(inliers) < .7:
        raise BowlRejected('RGB_tracking_inconsistent')
    scale = np.linalg.norm(affine[:, 0])
    if not .8 <= scale <= 1.2:
        raise BowlRejected('RGB_tracking_shape_change')
    result = cv2.warpAffine(np.asarray(mask, np.uint8), affine, current['bgr'].shape[1::-1], flags=cv2.INTER_NEAREST) > 0
    return result
