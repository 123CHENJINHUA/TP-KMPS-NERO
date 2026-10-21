"""Compare printed-board layouts using all matched marker corners, not IDs alone."""
import bootstrap
import cv2
import numpy as np


def marker_geometry(board, corners, ids):
    lookup = {int(n): np.asarray(p, np.float64)[:, :2] for n, p in zip(board.getIds().ravel(), board.getObjPoints())}
    obj, image, matched, sizes = [], [], [], []
    if ids is None:
        return {'matched_markers': 0, 'rms_px': None, 'max_px': None, 'rms_marker_fraction': None}
    for marker, points in zip(ids.ravel(), corners):
        if int(marker) not in lookup:
            continue
        points = np.asarray(points, np.float64).reshape(4, 2)
        obj.extend(lookup[int(marker)])
        image.extend(points)
        sizes.extend(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1))
        matched.append(int(marker))
    report = {'matched_markers': len(matched), 'matched_ids': sorted(matched),
              'rms_px': None, 'max_px': None, 'rms_marker_fraction': None}
    if len(set(matched)) < 4 or len(set(matched)) != len(matched):
        return report
    obj, image = np.asarray(obj, np.float64), np.asarray(image, np.float64)
    # Use EVERY matched corner, no RANSAC: a wrong board must not become a
    # plausible fit by throwing away all but one/few markers.
    H, _ = cv2.findHomography(obj, image, method=0)
    if H is None or not np.isfinite(H).all():
        return report
    predicted = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(predicted - image, axis=1)
    size = float(np.median(sizes))
    rms = float(np.sqrt(np.mean(errors ** 2)))
    report.update(rms_px=rms, max_px=float(errors.max()), median_marker_edge_px=size,
                  rms_marker_fraction=rms / max(size, 1e-8))
    return report


def layout_variants(config, dictionary):
    sx, sy = int(config['squares_x']), int(config['squares_y'])
    sizes = [(sx, sy), (sy, sx), (sx + 1, sy + 1), (sy + 1, sx + 1)]
    seen = set()
    for x, y in dict.fromkeys(sizes):
        for legacy in (bool(config.get('legacy_pattern', False)), not bool(config.get('legacy_pattern', False))):
            try:
                args = ((x, y), config['square_length_m'], config['marker_length_m'], dictionary)
                custom = config.get('marker_ids')
                base = cv2.aruco.CharucoBoard(*args) if custom is None else cv2.aruco.CharucoBoard(*args, np.array(custom, np.int32))
                base.setLegacyPattern(legacy)
            except cv2.error:
                continue
            ids = base.getIds().ravel()
            centers = np.array(base.getObjPoints()).mean(axis=1)
            column_ids = np.empty_like(ids)
            column_ids[np.lexsort((centers[:, 1], centers[:, 0]))] = np.sort(ids)
            for order, numbers in [('configured', ids), ('column-major', column_ids), ('reverse', ids[::-1])]:
                board = cv2.aruco.CharucoBoard(*args, np.array(numbers, np.int32))
                board.setLegacyPattern(legacy)
                key = (x, y, np.array(board.getObjPoints()).tobytes(), board.getIds().tobytes())
                if key in seen:
                    continue  # For odd rows the legacy flag gives identical geometry.
                seen.add(key)
                bc = dict(config, squares_x=x, squares_y=y, legacy_pattern=legacy, marker_ids=numbers.tolist())
                label = f'{x}x{y}, legacy={legacy}, ids={order}'
                yield label, board, bc, order


def diagnose_layout(image, config, dictionary, corners, ids):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    results = []
    for label, board, candidate, order in layout_variants(config, dictionary):
        cc = ci = None
        error = None
        try:
            cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray, markerCorners=corners, markerIds=ids)
        except cv2.error as exc:
            error = str(exc)  # Never silently convert an OpenCV failure into "0 corners".
        geometry = marker_geometry(board, corners, ids)
        count = 0 if ci is None else len(ci)
        expected = len(board.getIds())
        fit = geometry['rms_px']
        edge = geometry.get('median_marker_edge_px', 0.)
        complete = geometry['matched_markers'] == expected and expected == len(ids)
        good = bool(complete and count >= 10 and error is None and fit is not None and
                    fit <= max(1.5, .03 * edge) and geometry['max_px'] <= max(4., .10 * edge))
        results.append({'layout': label, 'board_config': candidate, 'marker_order': order,
                        'expected_markers': expected, 'matched_all_board_markers': complete,
                        'charuco_corners': count, 'expected_charuco_corners': (candidate['squares_x'] - 1) * (candidate['squares_y'] - 1),
                        'charuco_ids': [] if ci is None else ci.ravel().tolist(),
                        'charuco_corners_px': [] if cc is None else cc.reshape(-1, 2).tolist(),
                        'marker_geometry': geometry, 'opencv_error': error, 'plausible_full_board': good})
    results.sort(key=lambda r: (not r['plausible_full_board'],
                               r['marker_geometry']['rms_marker_fraction'] if r['marker_geometry']['rms_marker_fraction'] is not None else float('inf'),
                               -r['charuco_corners']))
    plausible = [r for r in results if r['plausible_full_board']]
    # A sole plausible fit is a suggestion for this image, not confirmation of
    # the board's physical millimetres or its generator's dictionary capacity.
    suggested = plausible[0]['board_config'] if len(plausible) == 1 else None
    return results, suggested


def draw_layout(image, corners, ids, layout=None):
    display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
    if layout and layout['charuco_corners']:
        cv2.aruco.drawDetectedCornersCharuco(display,
            np.array(layout['charuco_corners_px'], np.float32).reshape(-1, 1, 2),
            np.array(layout['charuco_ids'], np.int32).reshape(-1, 1))
    text = 'No plausible board layout' if not layout else layout['layout']
    cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    return display
