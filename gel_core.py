import os
import yaml
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import pandas as pd

# -------------------------
# Utilities
# -------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _rotate_bound(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(float(angle_deg)) < 1e-6: return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2.0) - (w / 2.0)
    M[1, 2] += (nH / 2.0) - (h / 2.0)
    return cv2.warpAffine(gray, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def estimate_rotation_deg(gray: np.ndarray) -> float:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=60, maxLineGap=20)
    if lines is None: return 0.0
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines[:, 0, :] if x2 != x1]
    angles = [a for a in angles if abs(a) <= 15]
    return float(np.median(angles)) if angles else 0.0

# -------------------------
# Image Processing Core
# -------------------------
def preprocess_image(path: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    color = cv2.imdecode(data, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    
    ang = estimate_rotation_deg(gray)
    color = _rotate_bound(color, -ang)
    gray = _rotate_bound(gray, -ang)

    blur = cv2.GaussianBlur(gray, (cfg.get("bg_subtract_ksize", 51), cfg.get("bg_subtract_ksize", 51)), 0)
    sub = cv2.subtract(gray, blur)
    
    # [NEW] 노이즈 제거 필터: 밴드의 선명도는 살리면서 배경의 자글거리는 노이즈를 뭉개줍니다.
    sub = cv2.bilateralFilter(sub, 9, 75, 75)
    
    p_high = np.percentile(sub, 99.5)
    if p_high > 10: 
        norm = np.clip(sub / p_high * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX)
    
    return norm, color

def find_lane_centers(row_img: np.ndarray, lane_total: int) -> list[int]:
    h, w = row_img.shape
    zone = row_img[int(h*0.2):int(h*0.8), :]
    prof = np.mean(zone, axis=0)
    
    margin = int(w * 0.04)
    prof[:margin] = 0
    prof[w-margin:] = 0
    
    prof_smooth = cv2.GaussianBlur(prof.reshape(1, -1), (1, 15), 0).ravel()
    peaks, _ = find_peaks(prof_smooth, distance=w//(lane_total+5), prominence=np.max(prof_smooth)*0.06)
    
    if len(peaks) < lane_total - 2:
        return [int(x) for x in np.linspace(margin, w - margin, lane_total)]
    
    return sorted(peaks)[:lane_total]

def analyze_ladder_1000bp(lane_img: np.ndarray) -> float:
    """
    Ladder lane에서 1000bp로 추정되는 y 위치 하나만 반환합니다.
    500bp는 더 이상 사용하지 않습니다.
    """
    h, w = lane_img.shape
    if np.max(lane_img) < 30:
        return float("nan")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 5))
    tophat = cv2.morphologyEx(lane_img, cv2.MORPH_TOPHAT, kernel).astype(np.float32)

    # well 근처 상단 잡음 제거
    tophat[:max(8, int(h * 0.12)), :] = 0

    prof = np.percentile(tophat, 95, axis=1).astype(np.float32)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 15), 0).ravel()

    prom = max(10.0, float(np.max(prof)) * 0.10)
    peaks, props = find_peaks(prof, prominence=prom, distance=8)

    if len(peaks) < 3:
        return float("nan")

    peaks = np.sort(peaks)

    # 100bp ladder에서 1000bp는 보통 상단부에서 가장 강하고 안정적인 밴드 중 하나이므로
    # 상단 55% 범위 내 peak들 중 prominence가 큰 peak를 선택
    upper_limit = int(h * 0.55)
    valid = [p for p in peaks if p <= upper_limit]

    if len(valid) == 0:
        return float("nan")

    best_peak = None
    best_score = -1.0
    for p in valid:
        y0 = max(0, p - 4)
        y1 = min(h, p + 5)
        score = float(np.sum(prof[y0:y1]))
        if score > best_score:
            best_score = score
            best_peak = p

    return float(best_peak) if best_peak is not None else float("nan")

def detect_band_2d(
    lane_img: np.ndarray,
    search_ranges: list[tuple[int, int]] | None = None,
    retry: bool = False,
    cfg: dict | None = None
) -> tuple[float, float, float, float]:
    h, w = lane_img.shape
    cfg = cfg or {}
    if h < 10 or w < 10:
        return float("nan"), float("nan"), 0.0, 0.0

    inner_margin = max(2, int(w * 0.18))
    x0 = inner_margin
    x1 = max(x0 + 1, w - inner_margin)
    core = lane_img[:, x0:x1]

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    core_eq = clahe.apply(core)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (core_eq.shape[1], 7))
    det = cv2.morphologyEx(core_eq, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    det = cv2.GaussianBlur(det, (3, 3), 0)

    if not search_ranges:
        search_ranges = [(int(h * 0.12), int(h * 0.86))]

    best_peak = None
    best_peak_score = -1.0

    for y_start, y_end in search_ranges:
        y_start = max(0, int(y_start))
        y_end = min(h, int(y_end))
        if y_end - y_start < 8:
            continue

        roi_global = det[y_start:y_end, :]
        if roi_global.size == 0:
            continue

        prof = np.percentile(roi_global, 97, axis=1).astype(np.float32)
        prof = gaussian_filter1d(prof, sigma=1.8)

        baseline = float(np.median(prof))
        mad = float(np.median(np.abs(prof - baseline))) + 1e-6
        z = (prof - baseline) / mad

        peak_thr = 1.8 if retry else 2.3
        peaks, _ = find_peaks(z, height=peak_thr, distance=8)
        if len(peaks) == 0:
            continue

        for p in peaks:
            p_abs = y_start + p

            local_half = max(8, int(h * 0.045))
            y0 = max(y_start, p_abs - local_half)
            y1 = min(y_end, p_abs + local_half + 1)
            roi = det[y0:y1, :]
            if roi.size == 0:
                continue

            roi_base = float(np.median(roi))
            roi_mad = float(np.median(np.abs(roi - roi_base))) + 1e-6
            thr = roi_base + (2.3 if retry else 2.8) * roi_mad

            bw = (roi > thr).astype(np.uint8) * 255
            bw = cv2.morphologyEx(
                bw,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            )
            bw = cv2.morphologyEx(
                bw,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
            )

            num, labels, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
            if num <= 1:
                continue

            for i in range(1, num):
                bx, by, bw_, bh_, area = stats[i]

                if area < 8:
                    continue
                if bw_ < 4:
                    continue
                if bw_ < bh_:
                    continue
                if bh_ > max(14, int(h * 0.09)):
                    continue

                mask_i = (labels == i)
                values = roi[mask_i]
                if values.size == 0:
                    continue

                integ = float(np.sum(values))
                mean_signal = float(np.mean(values))
                contrast = mean_signal - roi_base
                if contrast < 5:
                    continue

                center_l = int(roi.shape[1] * 0.30)
                center_r = int(roi.shape[1] * 0.70)

                center_region = roi[:, center_l:center_r]
                edge_region = np.hstack([roi[:, :center_l], roi[:, center_r:]]) if center_l > 0 and center_r < roi.shape[1] else roi

                center_mean = float(np.mean(center_region)) if center_region.size > 0 else 0.0
                edge_mean = float(np.mean(edge_region)) if edge_region.size > 0 else 0.0
                center_enrichment = center_mean - edge_mean

                min_center_enrichment = float(
                    cfg.get("min_center_enrichment_retry", 2.2) if retry
                    else cfg.get("min_center_enrichment_main", 4.0)
                )

                if center_enrichment < min_center_enrichment:
                    continue

                aspect = float(bw_) / max(1.0, float(bh_))
                cx_local = float(cent[i][0])
                center_penalty = abs(cx_local - (roi.shape[1] / 2.0))

                score = (
                    integ
                    * min(3.8, aspect)
                    * max(1.0, contrast / 10.0)
                    - center_penalty * 14.0
                )

                if score > best_peak_score:
                    best_peak_score = score
                    best_peak = (y0, roi, labels, i, x0)

    if best_peak is None:
        return float("nan"), float("nan"), 0.0, 0.0

    y0, roi, labels, best_idx, x_shift = best_peak
    mask = (labels == best_idx)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return float("nan"), float("nan"), 0.0, 0.0

    weights = roi[mask]
    wx = float(np.sum(xs * weights) / (np.sum(weights) + 1e-6))
    wy = float(np.sum(ys * weights) / (np.sum(weights) + 1e-6))

    integral = float(np.sum(weights))
    score = float(best_peak_score) / (float(core.shape[1]) * float(h) + 1e-6)

    return float(x_shift + wx), float(y0 + wy), integral, score

def get_recommendation(val: float, target: str, cfg: dict) -> str:
    rules = cfg.get("rules", {}).get(target, {})
    if not rules: return "X"
    
    thresholds = rules.get("thresholds", [])
    labels = rules.get("labels", [])
    
    for thr, lbl in zip(thresholds, labels):
        if val >= thr:
            return str(lbl)
    return "X"

def quantify_band_strength(lane_img: np.ndarray, band_y: float) -> float:
    h, w = lane_img.shape
    if np.isnan(band_y):
        return 0.0

    inner_margin = max(2, int(w * 0.15))
    x0 = inner_margin
    x1 = max(x0 + 1, w - inner_margin)
    core = lane_img[:, x0:x1].astype(np.float32)

    y = int(round(float(band_y)))
    y = max(0, min(h - 1, y))

    band_half = max(4, int(h * 0.018))
    bg_gap = max(3, int(h * 0.015))
    bg_half = max(5, int(h * 0.020))

    band_y0 = max(0, y - band_half)
    band_y1 = min(h, y + band_half + 1)
    band_roi = core[band_y0:band_y1, :]
    if band_roi.size == 0:
        return 0.0

    bg_parts = []

    up0 = max(0, y - band_half - bg_gap - bg_half)
    up1 = max(0, y - band_half - bg_gap)
    if up1 > up0:
        bg_parts.append(core[up0:up1, :])

    dn0 = min(h, y + band_half + bg_gap)
    dn1 = min(h, y + band_half + bg_gap + bg_half)
    if dn1 > dn0:
        bg_parts.append(core[dn0:dn1, :])

    if bg_parts:
        bg_roi = np.concatenate(bg_parts, axis=0)
        bg_level = float(np.median(bg_roi))
    else:
        bg_level = float(np.median(core))

    signal = np.clip(band_roi - bg_level, 0, None)

    if signal.size == 0:
        return 0.0

    peak = float(np.percentile(signal, 97))
    area = float(np.sum(signal))

    width_proxy = float(np.sum(signal > max(6.0, peak * 0.35)))
    mean_signal = float(np.mean(signal))

    score = (
        area * 0.75
        + peak * width_proxy * 2.2
        + mean_signal * band_roi.shape[1] * 1.0
    )

    return float(score)

def detect_band_profile_fallback(
    lane_img: np.ndarray,
    search_ranges: list[tuple[int, int]],
    z_threshold: float = 1.3,
    cfg: dict | None = None
) -> tuple[float, float, float, float]:
    h, w = lane_img.shape
    cfg = cfg or {}
    if h < 10 or w < 10:
        return float("nan"), float("nan"), 0.0, 0.0

    inner_margin = max(2, int(w * 0.15))
    x0 = inner_margin
    x1 = max(x0 + 1, w - inner_margin)
    core = lane_img[:, x0:x1]

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    core_eq = clahe.apply(core)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (core_eq.shape[1], 7))
    det = cv2.morphologyEx(core_eq, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    det = cv2.GaussianBlur(det, (3, 3), 0)

    best = None
    best_score = -1.0

    for y_start, y_end in search_ranges:
        y_start = max(0, int(y_start))
        y_end = min(h, int(y_end))
        if y_end - y_start < 6:
            continue

        roi = det[y_start:y_end, :]
        if roi.size == 0:
            continue

        prof = np.percentile(roi, 95, axis=1).astype(np.float32)
        prof = gaussian_filter1d(prof, sigma=1.2)

        base = float(np.median(prof))
        mad = float(np.median(np.abs(prof - base))) + 1e-6
        z = (prof - base) / mad

        peaks, _ = find_peaks(z, height=z_threshold, distance=6)
        if len(peaks) == 0:
            continue

        for p in peaks:
            y_abs = y_start + int(p)

            band_half = max(3, int(h * 0.015))
            y0 = max(0, y_abs - band_half)
            y1 = min(h, y_abs + band_half + 1)

            roi_band = det[y0:y1, :]
            if roi_band.size == 0:
                continue

            col_prof = np.mean(roi_band, axis=0)
            wx = float(np.argmax(col_prof))
            integral = float(np.sum(roi_band))
            peak_z = float(z[p])

            center_l = int(roi_band.shape[1] * 0.30)
            center_r = int(roi_band.shape[1] * 0.70)

            center_region = roi_band[:, center_l:center_r]
            edge_region = np.hstack([roi_band[:, :center_l], roi_band[:, center_r:]]) if center_l > 0 and center_r < roi_band.shape[1] else roi_band

            center_mean = float(np.mean(center_region)) if center_region.size > 0 else 0.0
            edge_mean = float(np.mean(edge_region)) if edge_region.size > 0 else 0.0
            center_enrichment = center_mean - edge_mean

            min_center_enrichment_fallback = float(cfg.get("min_center_enrichment_fallback", 1.8)) if cfg else 1.8
            if center_enrichment < min_center_enrichment_fallback:
                continue

            score = integral * max(1.0, peak_z / 2.0) * max(1.0, center_enrichment / 2.0)

            if score > best_score:
                best_score = score
                best = (x0 + wx, float(y_abs), integral, peak_z / 8.0)

    if best is None:
        return float("nan"), float("nan"), 0.0, 0.0

    return best

def score_target_candidate(
    rel_y: float,
    y1000_at_lane: float,
    strength: float,
    score: float,
    target_name: str,
    cfg: dict
) -> float:
    its_margin = float(cfg.get("its_below_1000_margin_px", 10))
    dead_zone = float(cfg.get("classification_dead_zone_px", 8))

    boundary = y1000_at_lane + its_margin

    if target_name == "16S":
        # 16S는 boundary보다 위쪽일수록 유리
        dist_penalty = max(0.0, rel_y - (boundary - dead_zone))
        target_bonus = 1.0 if rel_y <= boundary else 0.82
    else:
        # ITS는 boundary보다 아래쪽일수록 유리
        dist_penalty = max(0.0, (boundary + dead_zone * 0.6) - rel_y)
        target_bonus = 1.0 if rel_y >= boundary else 0.84

    combined = (
        float(strength) * target_bonus
        + float(score) * 15000.0
        - dist_penalty * 220.0
    )
    return float(combined)

def detect_band_multi_pass(
    row_img: np.ndarray,
    gray_width: int,
    cx: int,
    half_w: int,
    row_h: int,
    y1000_at_lane: float,
    target_mode: str,
    cfg: dict
) -> tuple[float, float, float, float, int, str, str]:
    detect_top_ignore_ratio = float(cfg.get("detect_top_ignore_ratio", 0.08))
    detect_bottom_ignore_ratio = float(cfg.get("detect_bottom_ignore_ratio", 0.05))

    valid_top = int(row_h * detect_top_ignore_ratio)
    valid_bottom = int(row_h * (1.0 - detect_bottom_ignore_ratio))

    its_max_row = int(row_h * float(cfg.get("its_max_row_ratio", 0.58)))
    fallback_max_row = int(row_h * float(cfg.get("fallback_max_row_ratio", 0.62)))

    search_16s = (
        max(valid_top, int(y1000_at_lane - float(cfg.get("band_search_16s_up_px", 34)))),
        min(valid_bottom, int(y1000_at_lane + float(cfg.get("band_search_16s_down_px", 16))))
    )

    search_its = (
        max(valid_top, int(y1000_at_lane + float(cfg.get("band_search_its_down_start_px", 14)))),
        min(valid_bottom, its_max_row, int(y1000_at_lane + float(cfg.get("band_search_its_down_end_px", 98))))
    )

    search_fallback = (
        max(valid_top, int(y1000_at_lane - float(cfg.get("band_search_fallback_up_px", 52)))),
        min(valid_bottom, fallback_max_row, int(y1000_at_lane + float(cfg.get("band_search_fallback_down_px", 125))))
    )

    if target_mode == "16S":
        target_candidates = [("16S", [search_16s])]
    elif target_mode == "ITS":
        target_candidates = [("ITS", [search_its])]
    else:
        target_candidates = [("16S", [search_16s]), ("ITS", [search_its])]

    best = None
    best_score = -1.0

    for dx in [0, -2, 2, -4, 4]:
        cx_try = int(cx + dx)
        x0 = max(0, cx_try - half_w)
        x1 = min(gray_width, cx_try + half_w)

        lane_img = row_img[:, x0:x1]
        if lane_img.size == 0:
            continue

        for target_name, search_ranges in target_candidates:
            rel_x, rel_y, _, score = detect_band_2d(
                lane_img,
                search_ranges=search_ranges,
                retry=False,
                cfg=cfg
            )
            method = "main"

            if np.isnan(rel_y):
                rel_x, rel_y, _, score = detect_band_2d(
                    lane_img,
                    search_ranges=[search_fallback],
                    retry=True,
                    cfg=cfg
                )
                method = "retry"

            if np.isnan(rel_y):
                rel_x, rel_y, _, score = detect_band_profile_fallback(
                    lane_img,
                    search_ranges=[search_fallback],
                    z_threshold=float(cfg.get("profile_fallback_z_threshold", 0.82)),
                    cfg=cfg
                )
                method = "fallback"

            if np.isnan(rel_y):
                continue

            strength = quantify_band_strength(lane_img, rel_y)

            # fallback은 더 엄격하게 채택
            if method == "fallback":
                if score < float(cfg.get("fallback_min_score", 0.020)):
                    continue
                if strength < float(cfg.get("fallback_min_strength", 4500)):
                    continue

            # target별 최소 강도
            if target_name == "16S" and strength < float(cfg.get("min_detect_strength_16s", 2500)):
                continue
            if target_name == "ITS" and strength < float(cfg.get("min_detect_strength_its", 3200)):
                continue

            combined_score = score_target_candidate(
                rel_y=rel_y,
                y1000_at_lane=y1000_at_lane,
                strength=strength,
                score=score,
                target_name=target_name,
                cfg=cfg
            )

            if combined_score > best_score:
                best_score = combined_score
                best = (x0 + rel_x, rel_y, strength, score, cx_try, target_name, method)

    if best is None:
        return float("nan"), float("nan"), 0.0, 0.0, cx, "UNK", "none"

    return best

# -------------------------
# Main Execution
# -------------------------
def analyze_gel(
    image_path: str,
    target_mode: str,
    config_path: str = "config.yaml",
    manual_anchors_1000: dict | None = None
) -> dict:
    cfg = load_config(config_path)
    gray, color = preprocess_image(image_path, cfg)

    h = gray.shape[0]
    n_rows = cfg["rows_to_split"]
    row_h = h // n_rows
    lane_count = cfg["lane_count_per_row"]
    half_w = cfg["lane_half_width_px"]

    results = []
    last_ab_1000 = None
    manual_anchors_1000 = manual_anchors_1000 or {}

    for r_idx in range(n_rows):
        row_y0, row_y1 = r_idx * row_h, (r_idx + 1) * row_h
        row_img = gray[row_y0:row_y1, :]
        centers = find_lane_centers(row_img, lane_count)

        ladder_1000_ys = []
        ladder_indices = []
        ladder_xs = []

        manual_anchor = manual_anchors_1000.get(r_idx + 1)

        if manual_anchor is not None:
            manual_lane = int(manual_anchor["lane"])
            manual_y = float(manual_anchor["y"])

            if 1 <= manual_lane <= len(centers):
                manual_cx = centers[manual_lane - 1]
                ladder_indices.append(manual_lane)
                ladder_xs.append(manual_cx)
                ladder_1000_ys.append(manual_y)
                a_1000, b_1000 = 0.0, manual_y
                last_ab_1000 = (float(a_1000), float(b_1000))
            else:
                if last_ab_1000 is not None:
                    a_1000, b_1000 = last_ab_1000
                else:
                    a_1000, b_1000 = 0.0, float(row_h * 0.35)

        else:
            fixed_ladder_lanes = cfg.get("ladder_lanes", [])

            for lane_idx, cx in enumerate(centers, start=1):
                if fixed_ladder_lanes and lane_idx not in fixed_ladder_lanes:
                    continue

                x0, x1 = max(0, cx - half_w), min(gray.shape[1], cx + half_w)
                lane_img = row_img[:, x0:x1]
                y_1000 = analyze_ladder_1000bp(lane_img)

                if not np.isnan(y_1000):
                    ladder_indices.append(lane_idx)
                    ladder_1000_ys.append(y_1000)
                    ladder_xs.append(cx)

            ladder_ok = (len(ladder_1000_ys) >= 2)
            if ladder_ok:
                if float(np.std(ladder_1000_ys)) > float(row_h) * 0.08:
                    ladder_ok = False

            if ladder_ok:
                a_1000, b_1000 = np.polyfit(
                    np.array(ladder_xs, dtype=np.float32),
                    np.array(ladder_1000_ys, dtype=np.float32),
                    1
                )
                last_ab_1000 = (float(a_1000), float(b_1000))
            else:
                if last_ab_1000 is not None:
                    a_1000, b_1000 = last_ab_1000
                else:
                    a_1000, b_1000 = 0.0, float(row_h * 0.35)

        fixed_ladder_lanes = cfg.get("ladder_lanes", [])

        for lane_idx, cx in enumerate(centers, start=1):
            x0, x1 = max(0, cx - half_w), min(gray.shape[1], cx + half_w)

            if target_mode == "AUTO" and lane_idx in fixed_ladder_lanes:
                y1000_at_lane = a_1000 * float(cx) + b_1000
                results.append({
                    "row": r_idx + 1,
                    "lane": lane_idx,
                    "target": "LADDER",
                    "center_x": cx,
                    "band_y": row_y0 + y1000_at_lane,
                    "band_integral": 0,
                    "recommend": "L",
                    "recommend_n": "L"
                })
                continue

            y1000_at_lane = a_1000 * float(cx) + b_1000
            its_margin = float(cfg.get("its_below_1000_margin_px", 8))

            abs_x, rel_y, strength, score, cx_used, target_guess, detect_method = detect_band_multi_pass(
                row_img=row_img,
                gray_width=gray.shape[1],
                cx=cx,
                half_w=half_w,
                row_h=row_h,
                y1000_at_lane=y1000_at_lane,
                target_mode=target_mode,
                cfg=cfg
            )

            if np.isnan(rel_y) or score < cfg.get("no_band_score_threshold", 0.010):
                abs_x = cx
                abs_y = ""
                label = "X"
                final_target = "UNK"
                strength = 0.0
            else:
                abs_y = row_y0 + rel_y

                if target_mode == "AUTO":
                    final_target = target_guess
                else:
                    final_target = target_mode

                label = get_recommendation(strength, final_target, cfg)

            results.append({
                "row": r_idx + 1,
                "lane": lane_idx,
                "target": final_target,
                "center_x": abs_x,
                "band_y": abs_y,
                "band_integral": strength,
                "recommend": label,
                "recommend_n": label
            })

    df = pd.DataFrame(results)
    df = normalize_strength_by_row(df)
    df = refine_recommendation_by_row(df)

    return {"df": df, "config": cfg, "color_img": color}

def normalize_strength_by_row(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "band_integral" not in out.columns:
        return out

    out["band_strength_norm"] = 0.0

    for row_num in sorted(out["row"].dropna().unique()):
        mask = (
            (out["row"] == row_num) &
            (out["target"] != "LADDER") &
            (out["target"] != "UNK") &
            (pd.to_numeric(out["band_integral"], errors="coerce") > 0)
        )

        vals = pd.to_numeric(out.loc[mask, "band_integral"], errors="coerce").dropna().values
        if len(vals) == 0:
            continue

        # 극단치 영향 완화
        ref = float(np.percentile(vals, 90))
        if ref <= 1e-6:
            ref = float(np.max(vals)) if len(vals) > 0 else 1.0
        ref = max(ref, 1.0)

        out.loc[mask, "band_strength_norm"] = (
            pd.to_numeric(out.loc[mask, "band_integral"], errors="coerce") / ref
        ).clip(lower=0.0, upper=2.0)

    return out

def refine_recommendation_by_row(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "band_strength_norm" not in out.columns:
        return out

    for idx, r in out.iterrows():
        target = str(r.get("target", ""))
        if target in ["LADDER", "UNK"]:
            continue

        norm = float(r.get("band_strength_norm", 0.0))
        strength = float(r.get("band_integral", 0.0))

        if target == "16S":
            if norm >= 0.78:
                rec = "3"
            elif norm >= 0.52:
                rec = "2"
            elif norm >= 0.24:
                rec = "1"
            elif norm > 0:
                rec = "0"
            else:
                rec = "X"

            # 절대강도 기반 상한
            if strength < 6000:
                rec = "0" if rec != "X" else "X"
            elif strength < 12000 and rec == "3":
                rec = "2"

        elif target == "ITS":
            if norm >= 1.28:
                rec = "20"
            elif norm >= 1.10:
                rec = "18"
            elif norm >= 0.94:
                rec = "16"
            elif norm >= 0.78:
                rec = "14"
            elif norm >= 0.60:
                rec = "12"
            elif norm >= 0.42:
                rec = "10"
            elif norm >= 0.26:
                rec = "8"
            elif norm > 0:
                rec = "6"
            else:
                rec = "X"

            if strength < 8500:
                rec = "6" if rec != "X" else "X"
            elif strength < 13000 and rec in ["20", "18", "16"]:
                rec = "12"
            elif strength < 17000 and rec == "20":
                rec = "16"

        else:
            rec = str(r.get("recommend", "X"))

        out.at[idx, "recommend"] = rec
        out.at[idx, "recommend_n"] = rec

    return out

def overlay_results(color_img: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    img = color_img.copy()

    for _, r in df.iterrows():
        cx = int(float(r["center_x"])) if pd.notna(r["center_x"]) else 0
        if cx <= 0:
            continue

        cy = int(float(r["band_y"])) if r["band_y"] != "" else -1
        if cy <= 0:
            continue

        target = str(r.get("target", ""))
        recommend = str(r.get("recommend", ""))

        if target == "LADDER":
            cv2.line(img, (cx - 30, cy), (cx + 30, cy), (0, 255, 255), 3)
            cv2.putText(
                img,
                "1000bp",
                (cx + 34, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            continue

        # 마커
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
        cv2.line(img, (cx, cy - 18), (cx, cy + 18), (0, 255, 0), 1)
        cv2.line(img, (cx - 12, cy), (cx + 12, cy), (0, 255, 0), 1)

        # 이미지에는 희석배수만 크게 표시
        label_text = recommend

        if target == "ITS":
            border_color = (0, 165, 255)   # orange
        elif target == "16S":
            border_color = (0, 255, 0)     # green
        else:
            border_color = (0, 0, 255)     # red

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.95
        thickness = 3

        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        tx = max(5, cx - tw // 2)
        ty = max(th + 12, cy - 18)

        box_x0 = max(0, tx - 8)
        box_y0 = max(0, ty - th - 8)
        box_x1 = min(img.shape[1] - 1, tx + tw + 8)
        box_y1 = min(img.shape[0] - 1, ty + baseline + 8)

        # 더 진한 반투명 검은 배경
        overlay = img.copy()
        cv2.rectangle(overlay, (box_x0, box_y0), (box_x1, box_y1), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)

        # 테두리
        cv2.rectangle(img, (box_x0, box_y0), (box_x1, box_y1), border_color, 2)

        # 검은 외곽선 + 흰 글자
        cv2.putText(
            img,
            label_text,
            (tx, ty),
            font,
            font_scale,
            (0, 0, 0),
            6,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            label_text,
            (tx, ty),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return img

def save_outputs(image_path: str, target: str, out_dir: str = None, config_path: str = "config.yaml"):
    out_dir = out_dir or os.path.dirname(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]
    
    analysis = analyze_gel(image_path, target, config_path)
    df, color_img = analysis["df"], analysis["color_img"]
    
    csv_path = os.path.join(out_dir, f"{base}_{target}_result.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    png_path = os.path.join(out_dir, f"{base}_{target}_result.png")
    cv2.imencode(".png", overlay_results(color_img, df))[1].tofile(png_path)
    
    return png_path, csv_path