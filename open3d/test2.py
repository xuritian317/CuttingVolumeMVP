import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("test2")
    # ply = o3d.io.read_point_cloud("main/box1.ply")
    # print(pcd)
    # print(np.asarray(pcd.points))
    # o3d.visualization.draw(ply)

    # octree
    N = 2000
    mesh = o3d.io.read_triangle_mesh("BunnyMesh.ply")
    o3d.visualization.draw_geometries([mesh])
    print(mesh.get_volume() * 1e6)

    pcd = mesh.sample_points_poisson_disk(N)
    # fit to unit cube
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
              center=pcd.get_center())
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    o3d.visualization.draw_geometries([pcd])

    print('octree division')
    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    o3d.visualization.draw_geometries([octree])
