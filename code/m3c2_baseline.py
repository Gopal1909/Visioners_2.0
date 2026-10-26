"""Bounded M3C2-style geometry; all coordinates, radii and uncertainties in metres.

Source-normal spherical PCA -> common finite cylinder in both epochs -> projected
mean difference (target minus source). No nearest-neighbour approximation. Unknown
predictions are -1. The runner never evaluates every point as a core.

Reference conventions: CloudCompare v2.13.2 qM3C2Process.cpp / qM3C2Tools.cpp,
mean mode, single-pass symmetric cylinder, population projected standard deviation.
This is not independently validated as equivalent to the reference application.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path
import platform
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.spatial import cKDTree
from c2c_baseline import (load_point_cloud_ply, compute_binary_metrics,
                          compute_combined_3class_metrics)


class Reason(IntEnum):
    VALID = 0
    NORMAL_INSUFFICIENT = 1
    NORMAL_DEGENERATE = 2
    NORMAL_UNSTABLE = 3
    SOURCE_SUPPORT = 4
    TARGET_SUPPORT = 8
    INVALID_UNCERTAINTY = 16
    RESOURCE_LIMIT = 32
    INVALID_NORMAL = 64


@dataclass(frozen=True)
class Config:
    normal_radii_m: tuple = (0.10, 0.20, 0.40)
    projection_radii_m: tuple = (0.05, 0.10, 0.20)
    half_length_m: float = 1.0
    min_normal_points: int = 10
    min_projected_points: int = 10
    max_curvature: float = 0.10
    min_eigengap: float = 0.10  # (lambda_middle-lambda_small)/lambda_large
    min_tangent_ratio: float = 0.05  # rejects line-like neighbourhoods
    max_stability_angle_deg: float = 30.0
    max_candidates: int = 250000  # reject oversized queries; never truncate them
    core_chunk_size: int = 128
    registration_error_m: float = 0.0
    minimum_effect_m: float = 0.02

    def __post_init__(self):
        for radii in (self.normal_radii_m, self.projection_radii_m):
            if not radii or not np.all(np.isfinite(radii)) or min(radii) <= 0:
                raise ValueError('Radii must be finite positive metres')
            if tuple(sorted(set(radii))) != tuple(radii):
                raise ValueError('Radii must be unique and increasing')
        if self.min_normal_points < 10 or self.min_projected_points < 10:
            raise ValueError('At least 10 normal and projected neighbours required')
        if not np.isfinite(self.half_length_m) or self.half_length_m <= 0:
            raise ValueError('Positive finite axial half-length required')
        if self.core_chunk_size < 1 or self.max_candidates < 10:
            raise ValueError('Invalid resource bounds')
        if not (0 <= self.max_curvature < 1 and 0 < self.min_eigengap <= 1
                and 0 < self.min_tangent_ratio <= 1
                and 0 < self.max_stability_angle_deg <= 90):
            raise ValueError('Invalid normal quality limits')
        # Invalid registration uncertainty is propagated as unknown by classify().


class MemoryTimer:
    """Scoped RSS sampler (10 ms) plus elapsed time; Windows lifetime peak separate."""
    def __enter__(self):
        self.process = psutil.Process()
        self.start_rss = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.t0 = time.perf_counter()
        def poll():
            while not self.stop.wait(0.01):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        info = self.process.memory_info()
        self.peak = max(self.peak, info.rss)
        self.stats = dict(seconds=time.perf_counter()-self.t0,
                          rss_start_mib=self.start_rss/2**20,
                          sampled_peak_rss_mib=self.peak/2**20,
                          sampled_rss_growth_mib=max(0, self.peak-self.start_rss)/2**20,
                          process_lifetime_peak_mib=getattr(info, 'peak_wset', 0)/2**20)


def bounded_ball(tree, center, radius, limit):
    """One bounded neighbour list at a time, with preallocation count guard."""
    count = tree.query_ball_point(center, radius, return_length=True)
    if count > limit:
        return None
    return np.asarray(tree.query_ball_point(center, radius), dtype=np.int64)


def estimate_normals(tree, cores, cfg=Config()):
    """Select lowest curvature among good PCA scales stable at another good scale.

    At least two adequate scales must agree within the configured unsigned angle.
    Largest absolute component positive; np.argmax breaks ties X, Y, then Z.
    All per-scale quality diagnostics are retained, including rejected scales.
    """
    m, k = len(cores), len(cfg.normal_radii_m)
    out = dict(normals=np.full((m, 3), np.nan), radius_m=np.full(m, np.nan),
               normal_reason=np.full(m, Reason.NORMAL_INSUFFICIENT, np.uint8),
               counts=np.zeros((m, k), np.int32), eigenvalues=np.full((m, k, 3), np.nan),
               curvature=np.full((m, k), np.nan), eigengap=np.full((m, k), np.nan),
               tangent_ratio=np.full((m, k), np.nan), adequate=np.zeros((m, k), bool),
               stability_angle_deg=np.full((m, k), np.nan),
               candidate_normals=np.full((m, k, 3), np.nan))
    for start in range(0, m, cfg.core_chunk_size):
        for i in range(start, min(start+cfg.core_chunk_size, m)):
            c = cores[i]
            if not np.all(np.isfinite(c)):
                out['normal_reason'][i] = Reason.INVALID_NORMAL
                continue
            ids = bounded_ball(tree, c, max(cfg.normal_radii_m), cfg.max_candidates)
            if ids is None:
                out['normal_reason'][i] = Reason.RESOURCE_LIMIT
                continue
            rel = tree.data[ids] - c
            d2 = np.einsum('ij,ij->i', rel, rel)
            for j, radius in enumerate(cfg.normal_radii_m):
                p = rel[d2 <= radius**2]
                out['counts'][i, j] = len(p)
                if len(p) < cfg.min_normal_points:
                    continue
                p = p - p.mean(axis=0)
                try:
                    values, vectors = np.linalg.eigh(p.T @ p / len(p))
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(values)) or values[-1] <= 1e-16:
                    continue
                values = np.maximum(values, 0)
                curv = values[0] / values.sum()
                gap = (values[1]-values[0])/values[2]
                ratio = values[1]/values[2]
                n = vectors[:, 0]
                if n[np.argmax(np.abs(n))] < 0:
                    n = -n
                out['eigenvalues'][i, j] = values
                out['curvature'][i, j] = curv
                out['eigengap'][i, j] = gap
                out['tangent_ratio'][i, j] = ratio
                out['candidate_normals'][i, j] = n
                out['adequate'][i, j] = (curv <= cfg.max_curvature and
                    gap >= cfg.min_eigengap and ratio >= cfg.min_tangent_ratio)
            good = np.flatnonzero(out['adequate'][i])
            stable = []
            for j in good:
                others = good[good != j]
                if len(others):
                    dot = np.abs(out['candidate_normals'][i, others] @ out['candidate_normals'][i, j])
                    angle = np.degrees(np.arccos(np.clip(dot, 0, 1))).min()
                    out['stability_angle_deg'][i, j] = angle
                    if angle <= cfg.max_stability_angle_deg:
                        stable.append(j)
            if stable:
                j = min(stable, key=lambda j: (out['curvature'][i, j], cfg.normal_radii_m[j]))
                out['normals'][i] = out['candidate_normals'][i, j]
                out['radius_m'][i] = cfg.normal_radii_m[j]
                out['normal_reason'][i] = Reason.VALID
            elif len(good):
                out['normal_reason'][i] = Reason.NORMAL_UNSTABLE
            elif out['counts'][i].max() >= cfg.min_normal_points:
                out['normal_reason'][i] = Reason.NORMAL_DEGENERATE
        if m >= 1000 and (start // cfg.core_chunk_size) % 4 == 0:
            print(f'  normals {min(start+cfg.core_chunk_size, m)}/{m}', flush=True)
    return out


def classify(raw, registration_error_m=0.0, minimum_effect_m=0.02):
    """Return LoD95, valid/reason and {-1 unknown,0 unchanged,1 changed}."""
    if not np.isfinite(minimum_effect_m) or minimum_effect_m < 0:
        raise ValueError('Minimum effect must be finite nonnegative metres')
    reason = raw['reason'].copy()
    lod = np.full(raw['signed_distance_m'].shape, np.nan)
    support = reason == 0
    if not np.isfinite(registration_error_m) or registration_error_m < 0:
        reason[support] = Reason.INVALID_UNCERTAINTY
    else:
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            lod[support] = 1.96 * (np.sqrt(
                raw['spread_source_m'][support]**2/raw['count_source'][support] +
                raw['spread_target_m'][support]**2/raw['count_target'][support]) + registration_error_m)
        bad = support & (~np.isfinite(lod) | (lod < 0) |
            ~np.isfinite(raw['signed_distance_m']) |
            ~np.isfinite(raw['spread_source_m']) | ~np.isfinite(raw['spread_target_m']) |
            (raw['spread_source_m'] < 0) | (raw['spread_target_m'] < 0))
        reason[bad] = Reason.INVALID_UNCERTAINTY
    valid = reason == 0
    pred = np.full(reason.shape, -1, np.int8)
    pred[valid] = (np.abs(raw['signed_distance_m'][valid]) >
                   np.maximum(lod[valid], minimum_effect_m)).astype(np.int8)
    lod[~valid] = np.nan
    return dict(lod95_m=lod, valid=valid, reason=reason, prediction=pred)


def project_direction(source_tree, target_tree, cores, normals, cfg=Config()):
    """Both epochs use exactly the same centered, symmetric finite cylinder.

    Query enclosing sphere once per core/epoch for all radii. Exact cylinder
    filtering follows. Oversized candidate spheres become RESOURCE_LIMIT unknowns.
    Counts/spreads refer to source/target order, not always epoch 0/1 order.
    """
    m, k = len(cores), len(cfg.projection_radii_m)
    shape = (m, k)
    raw = {key: np.full(shape, np.nan) for key in
           ('signed_distance_m', 'mean_source_m', 'mean_target_m',
            'spread_source_m', 'spread_target_m')}
    raw.update(count_source=np.zeros(shape, np.int32), count_target=np.zeros(shape, np.int32),
               reason=np.repeat(normals['normal_reason'][:, None], k, axis=1),
               candidate_count_source=np.zeros(m, np.int32),
               candidate_count_target=np.zeros(m, np.int32))
    bound = np.hypot(max(cfg.projection_radii_m), cfg.half_length_m)
    for start in range(0, m, cfg.core_chunk_size):
        for i in range(start, min(start+cfg.core_chunk_size, m)):
            if normals['normal_reason'][i] != Reason.VALID:
                continue
            n, c = normals['normals'][i], cores[i]
            if not np.all(np.isfinite(n)) or not np.isclose(np.linalg.norm(n), 1.0, atol=1e-8):
                raw['reason'][i] = Reason.INVALID_NORMAL
                continue
            for tag, tree in (('source', source_tree), ('target', target_tree)):
                ids = bounded_ball(tree, c, bound, cfg.max_candidates)
                if ids is None:
                    raw['reason'][i] |= Reason.RESOURCE_LIMIT
                    raw[f'candidate_count_{tag}'][i] = -1
                    continue
                raw[f'candidate_count_{tag}'][i] = len(ids)
                rel = tree.data[ids] - c
                axial = np.einsum('ij,j->i', rel, n)
                transverse2 = np.maximum(0, np.einsum('ij,ij->i', rel, rel)-axial**2)
                axial_ok = np.abs(axial) <= cfg.half_length_m
                for j, radius in enumerate(cfg.projection_radii_m):
                    t = axial[axial_ok & (transverse2 <= radius**2)]
                    raw[f'count_{tag}'][i, j] = len(t)
                    if len(t) >= cfg.min_projected_points:
                        raw[f'mean_{tag}_m'][i, j] = np.mean(t)
                        raw[f'spread_{tag}_m'][i, j] = np.std(t, ddof=0)
                    else:
                        raw['reason'][i, j] |= (Reason.SOURCE_SUPPORT if tag == 'source'
                                                else Reason.TARGET_SUPPORT)
            ok = raw['reason'][i] == 0
            raw['signed_distance_m'][i, ok] = raw['mean_target_m'][i, ok]-raw['mean_source_m'][i, ok]
        if m >= 1000 and (start // cfg.core_chunk_size) % 4 == 0:
            print(f'  cylinders {min(start+cfg.core_chunk_size, m)}/{m}', flush=True)
    return raw


def spatial_core_indices(xyz, count=2000, seed=42, tile_m=2.0):
    """Balanced round-robin over occupied 3D tiles; sample original points randomly.

    Every point contributes to tile inventory. Sparse occupied tiles get equal
    first-round weight. This is spatially balanced, not a point-uniform estimator.
    """
    if not 0 < count <= len(xyz) or tile_m <= 0:
        raise ValueError('Invalid sample size or tile size')
    origin = xyz.min(axis=0).astype(np.float64)
    dims = np.floor((xyz.max(axis=0)-origin)/tile_m).astype(np.int64)+1
    codes = np.empty(len(xyz), np.int64)
    for start in range(0, len(xyz), 100000):
        tiles = np.floor((xyz[start:start+100000]-origin)/tile_m).astype(np.int64)
        codes[start:start+len(tiles)] = (tiles[:, 0]*dims[1]+tiles[:, 1])*dims[2]+tiles[:, 2]
    order = np.argsort(codes, kind='stable')
    sorted_codes = codes[order]
    _, starts, sizes = np.unique(sorted_codes, return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    tile_order = rng.permutation(len(starts))
    allocation = np.zeros(len(starts), np.int32)
    remaining = count
    while remaining:
        for t in tile_order:
            if allocation[t] < sizes[t]:
                allocation[t] += 1
                remaining -= 1
                if remaining == 0:
                    break
    chosen = []
    for t in np.flatnonzero(allocation):
        offsets = rng.choice(sizes[t], size=allocation[t], replace=False)
        chosen.extend(order[starts[t]+offsets])
    indices = np.sort(np.asarray(chosen, np.int64))
    return indices, dict(seed=seed, tile_m=tile_m, tile_origin_m=origin.tolist(),
                         occupied_tiles=len(starts), sampled_tiles=int((allocation > 0).sum()),
                         rule='balanced occupied 3D tiles; random original points without replacement')


def score_pair(gt, predictions, masks):
    """Reuse Stage 2 metrics only after exact common-support masking."""
    row = {}
    for i in (0, 1):
        mask = masks[i]
        assert np.all(np.isin(predictions[i][mask], [0, 1]))
        met = compute_binary_metrics(gt[i][mask], predictions[i][mask])
        row.update({f'scan{i}_{key}': value for key, value in met.items()})
        row[f'scan{i}_total'] = len(mask)
        row[f'scan{i}_valid'] = int(mask.sum())
        row[f'scan{i}_coverage'] = float(mask.mean()) if len(mask) else 0.0
        row[f'scan{i}_support_excluded_rate'] = 1-row[f'scan{i}_coverage']
        row[f'scan{i}_unknown_rate'] = float(np.mean(predictions[i] < 0)) if len(mask) else 0.0
        row[f'scan{i}_correct_valid_over_total'] = float(
            np.sum(gt[i][mask] == predictions[i][mask])/max(1, len(mask)))
        for label in (0, 1):
            denom = int(np.sum(gt[i] == label))
            row[f'scan{i}_gt{label}_total'] = denom
            row[f'scan{i}_gt{label}_coverage'] = (
                float(np.sum(mask & (gt[i] == label))/denom) if denom else np.nan)
    mc = compute_combined_3class_metrics(gt[0][masks[0]], predictions[0][masks[0]],
                                        gt[1][masks[1]], predictions[1][masks[1]])
    row.update({f'combined_{key}': value for key, value in mc.items()
                if key not in ('confusion_matrix', 'class_names')})
    row['directional_macro_f1'] = (row['scan0_f1']+row['scan1_f1'])/2
    row['coverage'] = sum(int(m.sum()) for m in masks)/max(1, sum(len(m) for m in masks))
    row['support_excluded_rate'] = 1-row['coverage']
    row['unknown_rate'] = sum(int(np.sum(p < 0)) for p in predictions)/max(1, sum(len(m) for m in masks))
    row['conservative_accuracy_lower_bound'] = sum(
        int(np.sum(gt[i][masks[i]] == predictions[i][masks[i]])) for i in (0, 1)
    )/max(1, sum(len(m) for m in masks))
    return row


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def run_pilot(args):
    # Gate any dataset experiment on the synthetic suite, even on reruns.
    import unittest
    from test_m3c2_synthetic import GeometryTests
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests)
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        raise RuntimeError('Synthetic geometry gate failed; no dataset experiment started')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'synthetic_tests.json', dict(tests=test_result.testsRun, passed=True,
               failures=len(test_result.failures), errors=len(test_result.errors)))
    cfg = Config(registration_error_m=args.registration_error_m)
    config = dict(config=asdict(cfg), seed=args.seed, cores_per_scan=args.cores,
                  primary_projection_radius_m=0.10, primary_effect_floor_m=0.02,
                  effect_floors_m=[0.0, 0.02, 0.05], units='metres',
                  selection='A priori engineering configuration frozen before reading evaluation labels; no held-out claim',
                  scope='bounded clean within-pair exploratory pilot; no full-cloud core sweep',
                  reference_tag='CloudCompare v2.13.2', reference_revision='49dbbb6',
                  python=sys.version, executable=sys.executable, numpy=np.__version__,
                  scipy=scipy.__version__, platform=platform.platform(),
                  reason_codes={r.name: int(r) for r in Reason},
                  reason_note='normal codes 1/2/3 are exclusive; projection support bits 4/8/16/32/64 may combine',
                  command=sys.argv, timestamp_local='2026-10-25 Asia/Kolkata')
    write_json(out/'run_config.json', config)  # saved BEFORE label scoring
    write_json(out/'reference_conventions.json', dict(
        repository='https://github.com/CloudCompare/CloudCompare', tag='v2.13.2', revision='49dbbb6',
        process_source='https://github.com/CloudCompare/CloudCompare/blob/v2.13.2/plugins/core/Standard/qM3C2/src/qM3C2Process.cpp',
        statistics_source='https://github.com/CloudCompare/CloudCompare/blob/v2.13.2/plugins/core/Standard/qM3C2/src/qM3C2Tools.cpp',
        verified=['target mean minus source mean', 'population standard deviation (denominator n)',
                  '1.96 times (sqrt(sum of variance/count) plus registration RMS)',
                  'strict absolute distance greater than LoD',
                  'same core/normal/radius/half-length cylinder in both epochs',
                  'UI projection diameter divided by two; depth is half-length'],
        differences=['normal scale criterion', 'normal quality/stability rejection',
                     'explicit unknown policy', 'application minimum-effect floor',
                     'sphere plus exact cylinder filtering instead of octree cylinder query'],
        independent_reference_output_validation=False))
    timings = {}
    start = time.perf_counter()
    with MemoryTimer() as timer:
        data = [load_point_cloud_ply(p) for p in (args.cloud0, args.cloud1)]
        # Validate the explicit MultiChange3D element rather than assuming a vertex element.
        from plyfile import PlyData
        for p in (args.cloud0, args.cloud1):
            assert PlyData.read(p, mmap='r').elements[0].name == 'params'
        xyz = [d['xyz'] for d in data]
        for d in data:
            if not np.isfinite(d['xyz']).all() or not np.isin(d['cd_type'], [0, 1]).all():
                raise ValueError('Dataset coordinates/labels invalid')
    timings['load'] = timer.stats
    indices, sampling = [], []
    with MemoryTimer() as timer:
        for i in (0, 1):
            ids, meta = spatial_core_indices(xyz[i], args.cores, args.seed+i)
            indices.append(ids)
            sampling.append(meta)
    timings['sampling'] = timer.stats
    cores = [xyz[i][indices[i]].astype(np.float64) for i in (0, 1)]
    gt = [data[i]['cd_type'][indices[i]] for i in (0, 1)]
    np.savez_compressed(out/'selected_cores.npz', indices0=indices[0], indices1=indices[1],
                        xyz0=cores[0], xyz1=cores[1], cd_type0=gt[0], cd_type1=gt[1],
                        seeds=np.array([args.seed, args.seed+1]))
    write_json(out/'sampling.json', sampling)
    with MemoryTimer() as timer:
        trees = [cKDTree(p) for p in xyz]
    timings['shared_full_support_trees'] = timer.stats
    normals, raws = [], []
    for i in (0, 1):
        print(f'Scan {i}: normal estimation on full source support', flush=True)
        with MemoryTimer() as timer:
            normal = estimate_normals(trees[i], cores[i], cfg)
        timings[f'normal_scan{i}'] = timer.stats
        np.savez_compressed(out/f'normals_scan{i}.npz', **normal)
        print(f'Scan {i}: common cylinders on both full clouds', flush=True)
        with MemoryTimer() as timer:
            raw = project_direction(trees[i], trees[1-i], cores[i], normal, cfg)
        timings[f'projection_scan{i}'] = timer.stats
        np.savez_compressed(out/f'raw_scan{i}.npz', **raw)
        normals.append(normal)
        raws.append(raw)
    with MemoryTimer() as timer:
        c2c_dist = [trees[1-i].query(cores[i], k=1, workers=1)[0] for i in (0, 1)]
    timings['c2c_sample_queries'] = timer.stats
    c2c_pred = [(d > 0.075).astype(np.int8) for d in c2c_dist]
    np.savez_compressed(out/'c2c_same_cores.npz', distance0_m=c2c_dist[0],
                        distance1_m=c2c_dist[1], prediction0=c2c_pred[0], prediction1=c2c_pred[1])
    # Validate fresh queries against the historical archive on these exact indices.
    cache_validation = {}
    if Path(args.c2c_cache).exists():
        with np.load(args.c2c_cache) as cached:
            for i, key in enumerate(('distances_01', 'distances_10')):
                delta = np.max(np.abs(cached[key][indices[i]]-c2c_dist[i]))
                cache_validation[f'scan{i}_max_absolute_distance_difference_m'] = float(delta)
                if delta > 2e-6:
                    raise ValueError('Fresh C2C distances disagree with historical same-index cache')
                assert np.array_equal(cached[f'gt{i}'][indices[i]], gt[i])
    write_json(out/'c2c_cache_validation.json', cache_validation)
    rows, saved = [], {}
    for reg in dict.fromkeys((cfg.registration_error_m, 0.01, 0.02)):
        if reg != cfg.registration_error_m and reg in (0.01, 0.02):
            radii = (0.10,)
            floors = (0.02,)
        else:
            radii, floors = cfg.projection_radii_m, (0.0, 0.02, 0.05)
        for radius in radii:
            j = cfg.projection_radii_m.index(radius)
            for floor in floors:
                decisions = [classify(r, reg, floor) for r in raws]
                masks = [d['valid'][:, j] for d in decisions]
                preds = [d['prediction'][:, j] for d in decisions]
                params = dict(projection_radius_m=radius, effect_floor_m=floor,
                              registration_error_m=reg,
                              hypothetical_registration_sensitivity=reg != cfg.registration_error_m)
                for method, p in (('M3C2-style', preds), ('C2C common support', c2c_pred)):
                    rows.append(dict(method=method, **params, **score_pair(gt, p, masks)))
                if reg == cfg.registration_error_m:
                    key = f'r{radius:.2f}_f{floor:.2f}'
                    saved[key] = decisions
                    np.savez_compressed(out/f'predictions_{key}.npz',
                        **{f'scan{i}_{k}': v[:, j] for i, d in enumerate(decisions) for k, v in d.items()})
    all_masks = [np.ones(len(g), bool) for g in gt]
    rows.append(dict(method='C2C all sampled cores', projection_radius_m=np.nan,
                     effect_floor_m=np.nan, registration_error_m=np.nan,
                     hypothetical_registration_sensitivity=False,
                     **score_pair(gt, c2c_pred, all_masks)))
    pd.DataFrame(rows).to_csv(out/'configuration_metrics.csv', index=False)
    historical = pd.read_csv(Path(args.c2c_cache).parent/'threshold_metrics.csv')
    historical[np.isclose(historical.threshold_m, 0.075)].assign(
        scope='historical full-pair; not directly comparable to sampled M3C2').to_csv(
            out/'historical_c2c_full_pair.csv', index=False)
    # Half-length diagnostic on <=128 deterministic cores per epoch, not a sweep.
    from dataclasses import replace
    diagnostics = []
    for i in (0, 1):
        ids = np.linspace(0, args.cores-1, min(128, args.cores), dtype=int)
        subset_normals = {k: v[ids] for k, v in normals[i].items()}
        with MemoryTimer() as timer:
            half = project_direction(trees[i], trees[1-i], cores[i][ids], subset_normals,
                                     replace(cfg, half_length_m=0.5))
        timings[f'half_length_diagnostic_scan{i}'] = timer.stats
        np.savez_compressed(out/f'half_length_0.5_scan{i}.npz', positions_in_sample=ids, **half)
        j = cfg.projection_radii_m.index(0.10)
        valid = (half['reason'][:, j] == 0) & (raws[i]['reason'][ids, j] == 0)
        diff = np.abs(half['signed_distance_m'][valid, j]-raws[i]['signed_distance_m'][ids[valid], j])
        diagnostics.append(dict(scan=i, cores=len(ids), common_valid=int(valid.sum()),
                                median_absolute_distance_difference_m=float(np.median(diff)) if len(diff) else None,
                                max_absolute_distance_difference_m=float(diff.max()) if len(diff) else None))
    write_json(out/'half_length_diagnostic.json', diagnostics)
    timings['experiment_seconds_before_plotting'] = time.perf_counter()-start
    timings['m3c2_geometry_all_radii_seconds'] = sum(timings[key]['seconds'] for key in
        ('shared_full_support_trees', 'normal_scan0', 'normal_scan1', 'projection_scan0', 'projection_scan1'))
    timings['c2c_primary_search_seconds'] = (timings['shared_full_support_trees']['seconds']+
                                            timings['c2c_sample_queries']['seconds'])
    timings['scope_notes'] = ('Both search totals include the same shared tree build once. M3C2 projection '
        'time computes all three radii jointly; single-radius cost was not isolated. RSS is shared-process '
        'absolute memory with scoped sampled growth, not independent-process algorithm peak attribution.')
    write_json(out/'runtime_memory.json', timings)
    primary = saved['r0.10_f0.02']
    with MemoryTimer() as timer:
        make_plots(out, cores, gt, raws, primary, c2c_pred, cfg)
    timings['plotting'] = timer.stats
    timings['experiment_seconds_through_plotting'] = time.perf_counter()-start
    write_json(out/'runtime_memory.json', timings)
    from m3c2_report import generate_report
    generate_report(out)
    manifest = {str(p): dict(sha256=digest(p), bytes=Path(p).stat().st_size) for p in
                [args.cloud0, args.cloud1, Path(__file__), Path(__file__).with_name('test_m3c2_synthetic.py'),
                 Path(__file__).with_name('m3c2_report.py'),
                 Path(__file__).with_name('c2c_baseline.py'), Path(__file__).with_name('c2c_robustness.py')]}
    write_json(out/'input_source_manifest.json', manifest)
    print(pd.DataFrame(rows)[['method','projection_radius_m','effect_floor_m','registration_error_m',
                              'combined_macro_f1','combined_macro_mIoU','coverage']].to_string(index=False))
    print(f'Pilot saved: {out}', flush=True)


def make_plots(out, cores, gt, raws, decisions, c2c_pred, cfg):
    import matplotlib.pyplot as plt
    j = cfg.projection_radii_m.index(0.10)
    fig = plt.figure(figsize=(16, 10))
    for i in (0, 1):
        p = cores[i]
        pred = decisions[i]['prediction'][:, j]
        valid = decisions[i]['valid'][:, j]
        for col, (values, title) in enumerate(((gt[i], 'Ground truth'),
                (pred, 'M3C2-style: gray unknown / blue unchanged / red changed'),
                (np.where(valid, c2c_pred[i], -1), 'C2C on the same M3C2-valid cores'))):
            ax = fig.add_subplot(2, 3, i*3+col+1, projection='3d')
            colors = np.where(values < 0, '#9ca3af', np.where(values == 1, '#dc2626', '#2563eb'))
            ax.scatter(p[:,0], p[:,1], p[:,2], c=colors, s=3, alpha=.8)
            ax.set_title(f'Scan {i}: {title}', fontsize=9)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.set_box_aspect(np.maximum(np.ptp(p, axis=0), .1))
    fig.suptitle('2,000 spatially balanced original cores / epoch; full-cloud neighbourhood support\n'
                 'r=0.10 m, L=1.0 m, effect floor=0.02 m, registration uncertainty=0')
    fig.tight_layout()
    fig.savefig(out/'pilot_3d.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i in (0,1):
        valid = decisions[i]['valid'][:,j]
        d = raws[i]['signed_distance_m'][:,j]
        lod = decisions[i]['lod95_m'][:,j]
        axes[i,0].hist(d[valid], bins=70, color='#2563eb')
        axes[i,0].set(title=f'Scan {i}: valid signed distances', xlabel='target-source projection (m)', ylabel='cores')
        axes[i,1].scatter(lod[valid], np.abs(d[valid]), s=5, c=gt[i][valid], cmap='coolwarm')
        axes[i,1].plot([0, 1], [0, 1], '--', color='gray')
        axes[i,1].set(xlim=(0,.3), ylim=(0,1), title=f'Scan {i}: LoD and displacement (color = label)',
                      xlabel='LoD95 (m)', ylabel='absolute signed distance (m)')
    fig.tight_layout(); fig.savefig(out/'distance_uncertainty.png', dpi=160); plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true', help='synthetic tests only; no PLY loading')
    parser.add_argument('--cores', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--registration-error-m', type=float, default=0.0)
    base = root/'dataset/MultiChange3D/BikeParkingConstruction/pair_01'
    parser.add_argument('--cloud0', default=str(base/'pointCloud0.ply'))
    parser.add_argument('--cloud1', default=str(base/'pointCloud1.ply'))
    parser.add_argument('--c2c-cache', default=str(root/'results/pair_01/c2c_baseline/c2c_predictions.npz'))
    parser.add_argument('--output-dir', default=str(root/'results/pair_01/m3c2/pilot_2026-10-25_v1'))
    args = parser.parse_args()
    if args.smoke_test:
        import unittest
        from test_m3c2_synthetic import GeometryTests
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests))
        sys.exit(0 if result.wasSuccessful() else 1)
    if not 1 <= args.cores <= 20000:
        parser.error('Bounded runner permits 1..20000 cores per scan, never a full-cloud sweep')
    if not np.isfinite(args.registration_error_m) or args.registration_error_m < 0:
        parser.error('Registration uncertainty must be finite nonnegative metres')
    run_pilot(args)


if __name__ == '__main__':
    main()
