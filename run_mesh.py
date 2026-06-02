"""Standalone Poisson meshing for a dense point cloud (with normals + colours).

Mirrors the logic in mesh_handler.generate_poisson_mesh but works directly on a
saved .ply point cloud, so it does not require camera transforms (which the
gauss_to_pc.py meshing path otherwise mandates).
"""
import argparse
import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--depth", type=int, default=11)
    ap.add_argument("--laplacian_iters", type=int, default=5)
    ap.add_argument("--outlier_std", type=float, default=2.0)
    ap.add_argument("--density_quantile", type=float, default=0.07)
    ap.add_argument("--crop_min", type=float, nargs=3, default=None)
    ap.add_argument("--crop_max", type=float, nargs=3, default=None)
    args = ap.parse_args()

    print(f"Loading point cloud: {args.input}")
    pc = o3d.io.read_point_cloud(args.input)
    print(f"  points={len(pc.points)} normals={pc.has_normals()} colors={pc.has_colors()}")

    crop_box = None
    if args.crop_min is not None and args.crop_max is not None:
        crop_box = o3d.geometry.AxisAlignedBoundingBox(
            np.array(args.crop_min), np.array(args.crop_max))
        pc = pc.crop(crop_box)
        print(f"  cropped to {args.crop_min}..{args.crop_max}: points={len(pc.points)}")

    if not pc.has_normals():
        print("Estimating normals (missing in input)")
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pc.orient_normals_consistent_tangent_plane(30)

    print(f"Statistical outlier removal (nb=20, std_ratio={args.outlier_std})")
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=args.outlier_std)
    print(f"  points after cleaning={len(pc.points)}")

    print(f"Poisson surface reconstruction (depth={args.depth})")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=args.depth)
    print(f"  raw mesh: verts={len(mesh.vertices)} tris={len(mesh.triangles)}")

    densities = np.asarray(densities)
    thr = np.quantile(densities, args.density_quantile)
    mesh.remove_vertices_by_mask(densities < thr)
    print(f"  after density trim (<q{args.density_quantile}): "
          f"verts={len(mesh.vertices)} tris={len(mesh.triangles)}")

    # Poisson extrapolates a closed surface well beyond the points; clip the
    # balloon back to the point-cloud region.
    if crop_box is not None:
        mesh = mesh.crop(crop_box)
        print(f"  after mesh crop: verts={len(mesh.vertices)} tris={len(mesh.triangles)}")

    if args.laplacian_iters > 0:
        print(f"Laplacian smoothing ({args.laplacian_iters} iters)")
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=args.laplacian_iters,
            filter_scope=o3d.geometry.FilterScope.Vertex)

    mesh.compute_vertex_normals()
    print(f"Writing mesh: {args.output}")
    o3d.io.write_triangle_mesh(args.output, mesh)
    print("Done.")


if __name__ == "__main__":
    main()
