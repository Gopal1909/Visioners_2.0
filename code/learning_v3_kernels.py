"""Exact legacy geometry kernels without plotting/pandas imports.

Copies are checked by AST equality against the preserved source functions.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R_transform
RADII=(.15,.4,.8)
VOXELS=(.025,.05,.075)
K=64
MATCH_RADIUS=.25
SUPPORT_BOUND=max(r+np.sqrt(3)*v+MATCH_RADIUS for r,v in zip(RADII,VOXELS))

def raw_moments(relative,mask):
    n=mask.sum(1).clip(1);p=relative*mask[...,None]
    mean=p.sum(1)/n[:,None];q=(relative-mean[:,None])*mask[...,None]
    cov=np.einsum('nki,nkj->nij',q,q)/n[:,None,None]
    eig=np.linalg.eigvalsh(cov).clip(0);rad=np.linalg.norm(p,axis=2)
    return np.column_stack((mean,np.sqrt(np.diagonal(cov,axis1=1,axis2=2).clip(0)),eig,
        rad.max(1),rad.sum(1)/n,mask.sum(1)/K)).astype(np.float32)

def bounded_nearest(tree,points):
    distance,ids=tree.query(points,distance_upper_bound=MATCH_RADIUS,workers=1)
    valid=np.isfinite(distance)
    delta=np.zeros((*points.shape[:-1],3),np.float32)
    if valid.any():delta[valid]=(tree.data[ids[valid]]-points[valid])/MATCH_RADIUS
    return np.minimum(distance,MATCH_RADIUS).astype(np.float32)/MATCH_RADIUS,delta,valid

def coverage_patches(tree,centers,radius):
    """Keep core detail and actual full-radius context without a nearest-K cap."""
    points=np.zeros((len(centers),K,3),np.float64);valid=np.zeros((len(centers),K),bool)
    neighborhoods=tree.query_ball_point(centers,radius,workers=1,return_sorted=False)
    for row,ids in enumerate(neighborhoods):
        if not len(ids):continue
        p=tree.data[np.asarray(ids,np.int64)];delta=p-centers[row]
        d2=np.einsum('ij,ij->i',delta,delta)
        order=np.lexsort((p[:,2],p[:,1],p[:,0],d2))
        if len(order)>K:
            central=16
            ranks=np.r_[np.arange(central),central+np.linspace(0,len(order)-central-1,K-central).round().astype(int)]
            order=order[ranks]
        points[row,:len(order)]=p[order];valid[row,:len(order)]=True
    return points,valid

def features(trees,contexts,centers,chunk=256):
    """GT-free, translation-equivariant, bounded geometry for every channel."""
    n=len(centers);x=np.zeros((n,3,2,K,8),np.float32);mask=np.zeros((n,3,2,K),bool)
    geometry=[]
    core_distance,core_delta,core_valid=bounded_nearest(trees[1],centers)
    for scale,radius in enumerate(RADII):
        stats=[]
        for side in (0,1):
            tree=contexts[scale][side]
            for start in range(0,n,chunk):
                end=min(n,start+chunk);c=centers[start:end]
                points,valid=coverage_patches(tree,c,radius)
                relative=(points-c[:,None,:])/radius
                d,delta,matched=bounded_nearest(trees[1-side],points.reshape(-1,3))
                channels=np.concatenate((relative,delta.reshape(-1,K,3),d.reshape(-1,K,1),matched.reshape(-1,K,1)),axis=-1)
                channels[~valid]=0.;x[start:end,scale,side]=channels;mask[start:end,scale,side]=valid
            stats.append(raw_moments(x[:,scale,side,:,:3],mask[:,scale,side]))
        counts=[np.log1p(t.query_ball_point(centers,radius,return_length=True,workers=1))/np.log1p(10000) for t in trees]
        a,b=stats
        geometry.append(np.column_stack((a,b,b-a,np.abs(b-a),*counts,core_distance,core_delta,core_valid)))
    geo=np.concatenate(geometry,axis=1).astype(np.float32)
    assert geo.shape==(n,165) and np.isfinite(x).all() and np.isfinite(geo).all()
    return dict(x=x,mask=mask,geo=geo)

def discover_pair(base, pair):
    folder = Path(base)/f'pair_{pair}'
    files = []
    for scan in (0,1):
        found = sorted(folder.rglob(f'pointCloud{scan}.ply'))
        if len(found) != 1:
            raise ValueError(f'{folder}: expected one cloud {scan}, found {len(found)}')
        files.append(found[0])
    return files

def cell_coordinates(xyz, protocol):
    local = (np.asarray(xyz)[:,:2].astype(np.float64)-protocol['origin_xy_m'])/protocol['cell_size_m']
    cell = np.floor(local).astype(np.int64)
    inside = (local-cell)*protocol['cell_size_m']
    guard = protocol['core_boundary_guard_m']
    eligible = ((inside >= guard)&(inside <= protocol['cell_size_m']-guard)).all(1)
    return cell, eligible

def split_mask(xyz, protocol, split):
    cell, eligible = cell_coordinates(xyz,protocol)
    selected = np.zeros(len(xyz),bool)
    for x,y in protocol['cells'][split]:
        selected |= (cell[:,0] == x)&(cell[:,1] == y)
    return eligible&selected

def apply_rigid_transformation(xyz, translation_m, rotation_deg, center=None, seed=42):
    """
    Apply controlled rigid body rotation and translation to a copy of point coordinates.
    Rotation is applied around the cloud centroid.
    """
    xyz_perturbed = xyz.copy()
    if translation_m == 0.0 and rotation_deg == 0.0:
        return xyz_perturbed

    if center is None:
        center = np.mean(xyz, axis=0)

    rng = np.random.default_rng(seed)

    # Deterministic rotation axis
    if rotation_deg != 0.0:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rot = R_transform.from_rotvec(np.radians(rotation_deg) * axis)
        R_matrix = rot.as_matrix().astype(np.float32)

        xyz_centered = xyz_perturbed - center
        xyz_rotated = (xyz_centered @ R_matrix.T) + center
        xyz_perturbed = xyz_rotated

    # Deterministic translation vector
    if translation_m != 0.0:
        t_dir = rng.normal(size=3)
        t_dir /= np.linalg.norm(t_dir)
        translation_vec = (translation_m * t_dir).astype(np.float32)
        xyz_perturbed = xyz_perturbed + translation_vec

    return xyz_perturbed

def apply_point_sparsity(xyz, labels, retain_ratio, seed=42):
    """Randomly retain a fraction of points and their corresponding labels."""
    if retain_ratio >= 1.0:
        return xyz.copy(), labels.copy()

    rng = np.random.default_rng(seed)
    n_pts = len(xyz)
    n_retain = int(np.round(n_pts * retain_ratio))
    indices = rng.choice(n_pts, size=n_retain, replace=False)
    indices.sort()
    return xyz[indices].copy(), labels[indices].copy()
