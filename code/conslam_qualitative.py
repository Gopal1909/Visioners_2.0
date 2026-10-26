"""Bounded, headless ConSLAM inspection without change-ground-truth claims."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from m3c2_baseline import write_json,digest


def run():
    import open3d as o3d
    root=Path(__file__).resolve().parents[1]
    out=root/'results/conslam_qualitative_2026-10-25_v2'
    out.mkdir(parents=True,exist_ok=True)
    base=root/'dataset/Sequence 3'
    pcdfolder=base/'Copy of lidar-pcd/lidar-pcd'
    posefolder=base/'Copy of lidar_pose_normalized/normalized'
    files=sorted(pcdfolder.glob('*.pcd'))
    paired=[f for f in files if (posefolder/(f.stem+'.txt')).exists()]
    if len(paired)<2:raise ValueError('Need two timestamp-matched LiDAR poses')
    selected=[paired[0],paired[1]]  # first two timestamp-matched poses, 4.84 s apart
    local=[];registered=[];poses=[];manifest=[]
    for p in selected:
        cloud=o3d.io.read_point_cloud(str(p))
        xyz=np.asarray(cloud.points)
        xyz=xyz[np.isfinite(xyz).all(1)]
        posefile=posefolder/(p.stem+'.txt')
        transform=np.loadtxt(posefile)
        assert transform.shape==(4,4) and np.isfinite(transform).all()
        np.testing.assert_allclose(transform[3],[0,0,0,1],atol=1e-5)
        np.testing.assert_allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-4)
        local.append(xyz)
        registered.append(xyz@transform[:3,:3].T+transform[:3,3])
        poses.append(transform)
        manifest.append(dict(pcd=str(p),pose=str(posefile),points=len(xyz),
            pcd_bytes=p.stat().st_size,pcd_sha256=digest(p),pose_sha256=digest(posefile),
            pose_matrix=transform.tolist()))
    tlsfile=base/'Copy of groundtruth_scan.ply'
    tls=PlyData.read(str(tlsfile),mmap='r').elements[0].data
    ids=np.linspace(0,len(tls)-1,min(50000,len(tls)),dtype=np.int64)
    tlsxyz=np.column_stack([tls[k][ids] for k in ('x','y','z')])
    tlsxyz=tlsxyz[np.isfinite(tlsxyz).all(1)]
    query=np.linspace(0,len(registered[0])-1,min(5000,len(registered[0])),dtype=int)
    dist=cKDTree(registered[1]).query(registered[0][query],workers=1)[0]
    rawdist=cKDTree(local[1]).query(local[0][query],workers=1)[0]
    # These are geometric proximity summaries, never labelled accuracy measures.
    summary=dict(sequence=3,total_pcd_files=len(files),selected_frames=manifest,
        timestamp_matched_frame_count=len(paired),frame_index_gap=files.index(selected[1])-files.index(selected[0]),
        time_gap_seconds=(int(selected[1].stem)-int(selected[0].stem))/1e6,
        tls_points=len(tls),tls_sampled_points=len(tlsxyz),tls_bytes=tlsfile.stat().st_size,
        tls_element_fields=list(tls.dtype.names),
        median_nn_raw_frame_m=float(np.median(rawdist)),median_nn_pose_applied_m=float(np.median(dist)),
        sampled_scan0_fraction_distance_above_0_075_m=float(np.mean(dist>.075)),
        transform_convention='Apply normalized 4x4 pose as local-to-normalized-frame; no independent TLS registration check',
        limitations=['One acquisition sequence, not a cross-month construction comparison',
                    'No official per-point change labels; no accuracy/F1/IoU or learned generalization score',
                    'Proximity changes can arise from viewpoint/coverage, registration and motion',
                    'TLS coordinate relation to normalized LiDAR poses is unverified; TLS shown separately'],
        citation='Trzeciak et al. (2023), ConSLAM: Construction Data Set for SLAM, doi:10.1061/JCCEE5.CPENG-5212')
    write_json(out/'inspection.json',summary)
    fig=plt.figure(figsize=(17,5.5))
    ax=fig.add_subplot(1,3,1,projection='3d')
    ax.scatter(tlsxyz[:,0],tlsxyz[:,1],tlsxyz[:,2],c=tlsxyz[:,2],cmap='viridis',s=.3)
    ax.set_title('TLS site context\n50,000 original sample points; separate frame',fontsize=10)
    for i,(xyz,color) in enumerate(zip(registered,['#2563eb','#ea580c'])):
        ax2=fig.add_subplot(1,3,2,projection='3d') if i==0 else ax2
        ii=np.linspace(0,len(xyz)-1,min(10000,len(xyz)),dtype=int)
        ax2.scatter(xyz[ii,0],xyz[ii,1],xyz[ii,2],c=color,s=.5,label=f'LiDAR frame {i}')
    ax2.legend(fontsize=8);ax2.set_title('Two local sweeps with supplied poses\nWithin one acquisition sequence',fontsize=10)
    ax3=fig.add_subplot(1,3,3,projection='3d');xyz=registered[0][query]
    ax3.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=np.where(dist>.075,'#dc2626','#2563eb'),s=1)
    ax3.set_title('C2C geometric discrepancy >7.5 cm (red)\nNo verified physical-change labels',fontsize=10)
    for a in (ax,ax2,ax3):
        a.set_xlabel('X (m)');a.set_ylabel('Y (m)');a.set_zlabel('Z (m)')
    fig.suptitle('ConSLAM qualitative context only: coverage and proximity are not construction progress')
    fig.tight_layout();fig.savefig(out/'conslam_qualitative.png',dpi=170);plt.close(fig)
    # Trajectory inventories all available poses without assembling a dense map.
    trajectory=[]
    for f in sorted(posefolder.glob('*.txt')):
        t=np.loadtxt(f)
        if t.shape==(4,4):trajectory.append(t[:3,3])
    trajectory=np.asarray(trajectory)
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(trajectory[:,0],trajectory[:,1],lw=1,color='#2563eb')
    ax.scatter([p[0,3] for p in poses],[p[1,3] for p in poses],c=['#2563eb','#ea580c'],s=45)
    ax.set(title='ConSLAM Sequence 3 normalized trajectory',xlabel='X (m)',ylabel='Y (m)');ax.axis('equal');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'conslam_trajectory.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='selected_frames'},indent=2))


if __name__=='__main__':run()
