"""RGB outline + known 50mm cube on a registered support plane.

No cube-surface depth is read. This is conditional upright-on-table geometry,
not monocular 6D tracking or evidence of table contact. Full visible outlines
are required; failed/occluded frames are never relabelled as fresh.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

from .bowl_image import basis, project, resample, boundary_distances


class CubeRejected(ValueError):
    def __init__(self, reason, checks=None):
        super().__init__(reason)
        self.reason, self.checks = reason, checks or {}


def cube_vertices(centre, yaw, half, normal):
    u, v = basis(normal)
    x = np.cos(yaw)*u+np.sin(yaw)*v
    y = -np.sin(yaw)*u+np.cos(yaw)*v
    rotation = np.column_stack([x, y, normal])
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])*half
    return corners @ rotation.T+centre, rotation


def cube_silhouette(centre, yaw, half, normal, K, transform):
    corners, _ = cube_vertices(centre, yaw, half, normal)
    return cv2.convexHull(project(corners, K, transform).astype(np.float32)).reshape(-1, 2).astype(float)


def estimate_cube(definition, mask, K, transform, *, table_normal, table_offset, rgb, initial=None):
    """Return base-centred cube pose, local support height, and image checks."""
    mask = np.asarray(mask, np.uint8)
    half = np.asarray(definition['size']['half_extents'], float)
    normal = np.asarray(table_normal, float)
    if (mask.ndim != 2 or rgb.shape[:2] != mask.shape or normal.shape != (3,)
            or not np.isfinite(normal).all() or not np.isfinite(table_offset)
            or abs(np.linalg.norm(normal)-1) > 1e-4 or normal[2] < np.cos(np.deg2rad(5.))
            or half.shape != (3,) or not np.allclose(half, .025, atol=1e-6)):
        raise CubeRejected('cube_input_or_plane_invalid')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise CubeRejected('cube_mask_missing')
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    convexity = area/max(cv2.contourArea(hull), 1)
    if area < 120 or convexity < .85:
        raise CubeRejected('cube_outline_occluded_or_nonconvex',
                           dict(mask_convexity=float(convexity), mask_area_px=float(area)))
    # A narrow shadow or colour-threshold notch can break the otherwise full
    # cube outline. Recover only small concavities, then demand current RGB
    # support, 5cm reprojection agreement and temporal confirmation below.
    minor_concavity_filled = convexity < .94
    if minor_concavity_filled:
        contour = hull
    xy = contour.reshape(-1, 2)
    height, width = mask.shape
    if (xy[:, 0].min() <= 2 or xy[:, 1].min() <= 2
            or xy[:, 0].max() >= width-3 or xy[:, 1].max() >= height-3):
        raise CubeRejected('cube_mask_clipped')
    polygon = cv2.approxPolyDP(contour, .65, True).reshape(-1, 2).astype(float)
    observed = resample(polygon, 40)
    u, v = basis(normal)
    plane_point = (table_offset+half[2])*normal
    pixel = np.r_[np.mean(observed, axis=0), 1.]
    ray = (np.linalg.inv(K) @ pixel) @ transform[:3, :3].T
    distance = (table_offset+half[2]-normal @ transform[:3, 3])/(normal @ ray)
    if not np.isfinite(distance) or not .1 < distance < 2.:
        raise CubeRejected('cube_ray_invalid')
    seed = transform[:3, 3]+distance*ray
    seed_uv = np.array([u @ seed, v @ seed])

    def model(parameters):
        centre = plane_point+parameters[0]*u+parameters[1]*v
        return cube_silhouette(centre, parameters[2], half, normal, K, transform)

    def residual(parameters):
        outline = model(parameters)
        return np.r_[boundary_distances(observed, outline),
                     boundary_distances(resample(outline, 40), polygon)]

    if initial is None:
        starts = [np.r_[seed_uv, angle] for angle in (-.6, 0., .6)]
    else:
        position, yaw = initial
        prior = np.array([u @ position, v @ position, yaw])
        if np.any(np.abs(prior[:2]-seed_uv) >= .055):
            starts = [np.r_[seed_uv, angle] for angle in (-.6, 0., .6)]
        else:
            starts = [prior]
    best = None
    for start in starts:
        result = least_squares(residual, start,
                               bounds=(np.r_[seed_uv-.06, -2*np.pi], np.r_[seed_uv+.06, 2*np.pi]),
                               x_scale=[.05, .05, .5], diff_step=1e-3,
                               max_nfev=18, ftol=1e-4, xtol=1e-5, gtol=1e-4)
        cost = float(np.mean(residual(result.x)**2))
        if best is None or cost < best[0]:
            best = cost, result.x
    _, parameters = best
    centre = plane_point+parameters[0]*u+parameters[1]*v
    yaw = float((parameters[2]+np.pi/4) % (np.pi/2)-np.pi/4)
    outline = model(parameters)
    # Work in the small object ROI for pixel IoU and current-colour support.
    x, y, w, h = cv2.boundingRect(np.vstack([polygon, outline]).astype(np.float32))
    x, y = max(0, x-3), max(0, y-3)
    w, h = min(width-x, w+6), min(height-y, h+6)
    predicted, measured = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(predicted, np.rint(outline-[x, y]).astype(np.int32), 1)
    cv2.fillPoly(measured, [np.rint(polygon-[x, y]).astype(np.int32)], 1)
    iou = np.count_nonzero(predicted & measured)/max(np.count_nonzero(predicted | measured), 1)
    p95 = float(np.percentile(residual(parameters), 95))
    # The candidate must correspond to today's green pixels, not a retained mask.
    hsv = cv2.cvtColor(rgb[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 100) & (hsv[..., 2] >= 25)
    colour_support = float(np.mean(green[predicted.astype(bool)]))
    checks = dict(method='RGB_known_cube_support_plane', cube_surface_depth_used=False,
                  upright_on_table_assumed=True, pose_scope='conditional_upright_on_table_not_full_6D',
                  height_source='support_plane_plus_known_half_side', table_contact_verified=False,
                  silhouette_iou=float(iou), outline_p95_px=p95, green_support=colour_support,
                  table_normal_base=normal.tolist(), table_plane_offset_m=float(table_offset),
                  known_side_m=.05, yaw_rad=yaw, fit_centre_base_m=centre.tolist(),
                  mask_convexity=float(convexity), minor_concavity_filled=minor_concavity_filled)
    # A real 50 mm cube near the lower image edge gives about 0.85 IoU,
    # 3-4 px outline residual and 0.87 current-green support. Keep these
    # borderline frames only after three consistent RGB measurements.
    checks['requires_temporal_confirmation'] = bool(minor_concavity_filled or p95 > 2.2 or iou < .90 or colour_support < .90)
    if iou < .82 or p95 > 4.5:
        raise CubeRejected('cube_known_shape_mismatch', checks)
    if colour_support < .85:
        raise CubeRejected('cube_current_RGB_not_supported', checks)
    pose = np.eye(4)
    pose[:3, 3] = centre
    pose[:3, :3] = cube_vertices(centre, yaw, half, normal)[1]
    bottom = centre-half[2]*normal
    return pose, float(bottom[2]), checks


def green_mask(frame, seed_mask, hue=None):
    """Local colour segmentation measured anew in the current RGB frame."""
    seed = np.asarray(seed_mask, bool)
    ys, xs = np.nonzero(seed)
    if len(ys) < 120:
        raise CubeRejected('cube_seed_mask_insufficient')
    height, width = seed.shape
    x0, x1 = max(0, xs.min()-35), min(width, xs.max()+36)
    y0, y1 = max(0, ys.min()-35), min(height, ys.max()+36)
    hsv = frame.get('hsv')
    if hsv is None:
        hsv = cv2.cvtColor(frame['bgr'], cv2.COLOR_BGR2HSV)
    local = hsv[y0:y1, x0:x1]
    selected = ((local[..., 0] >= 30) & (local[..., 0] <= 100)
                & (local[..., 1] >= 100) & (local[..., 2] >= 25))
    if hue is not None:
        delta = np.abs(local[..., 0].astype(float)-hue)
        selected &= np.minimum(delta, 180-delta) <= 14
    selected = cv2.morphologyEx(selected.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected)
    if count <= 1:
        raise CubeRejected('cube_current_colour_missing')
    # Associate to the SAM/last measured location; never jump to another green item.
    overlaps = np.bincount(labels[seed[y0:y1, x0:x1]], minlength=count)
    overlaps[0] = 0
    label = int(np.argmax(overlaps))
    if overlaps[label] < .25*seed.sum() or not .55*seed.sum() <= stats[label, cv2.CC_STAT_AREA] <= 1.8*seed.sum():
        raise CubeRejected('cube_colour_association_failed')
    mask = np.zeros(seed.shape, bool)
    mask[y0:y1, x0:x1] = labels == label
    return mask
