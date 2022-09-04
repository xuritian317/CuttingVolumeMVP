import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. intel box 11.5*11.5*6 = 793.5             744
    # 2. ten book 14.5*20.7*1.3 = 390.195          313  downpcd.segment_plane(distance_threshold=0.005
        # ten2                 396  392         downpcd.segment_plane(distance_threshold=0.005
    # 3. noah book 8.5*11*0.45(0.4-0.5)= 42.075
    # 4. cp shelf 26(25-26)*5.2*2.5 = 338          146  143
    # 5. ten badge 3(2.4-3.6)*5.0*5.6 = 84         39 39

    print("try to calculate the volume of cutting ... ")

    print("->正在加载点云... ")
    pcd = o3d.io.read_point_cloud("../src/cutting1.ply")
    print(pcd)
    # print(np.asarray(pcd.points))
    # o3d.io.write_point_cloud("box1.pcd", pcd)

    # print("->正在可视化点云")
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw([pcd])


    # 点云裁切
    # vol = o3d.visualization.read_selection_polygon_volume("cropped.json")
    # box = vol.crop_point_cloud(pcd)

    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # box.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # print("Diplaying cropped pointcloud")
    # o3d.visualization.draw([pcd])
    # o3d.visualization.draw([box])


    # 点云下采样
    downpcd = pcd  # .uniform_down_sample(every_k_points=1)
    # downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    # o3d.visualization.draw([pcd])
    o3d.visualization.draw([downpcd])

    # 平面切割 获得多目标(立于平面之上)
    plane_model, inliers = downpcd.segment_plane(distance_threshold=0.003,
                                                 ransac_n=3,
                                                 num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # inlier_cloud = downpcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = downpcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw([outlier_cloud])

    # 同类聚集
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    print(labels)

    # 根据聚集结构构造主要目标
    ori_pc = np.asarray(outlier_cloud.points)
    print(ori_pc.size)

    obj1xyz = []
    tLabel = labels.max()
    # print(type(obj1xyz))
    for pc, l in zip(ori_pc, labels):
        # x, y, z = pc
        if l == tLabel:
            obj1xyz.append(pc)

    obj1xyz = np.array(obj1xyz)
    # print(obj1xyz)
    print(obj1xyz.size)

    obj1 = o3d.geometry.PointCloud()
    obj1.points = o3d.utility.Vector3dVector(obj1xyz)
    o3d.visualization.draw_geometries([obj1])

    # 点云凸包计算
    hull, _ = obj1.compute_convex_hull()  # [hull]: 739.505592058634
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries([hull_ls])

    print(f"[hull]: {hull.get_volume() * 1e6}")  # 746.518 CM^2

    # for pc, l in zip(ori_pc, labels):
    #     # x, y, z = pc
    #     print(pc, l)

    # 更换颜色
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([outlier_cloud])

    # pointcloud 转 mesh
    obj1.estimate_normals()
    # estimate radius for rolling ball
    distances = obj1.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     obj1,
    #     o3d.utility.DoubleVector([radius, radius * 2]))
    # o3d.visualization.draw_geometries([mesh])

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        obj1,
        alpha=0.01,
    )
    o3d.visualization.draw_geometries([mesh])
    # print(mesh.get_volume() * 1e6)  # The mesh is not watertight, and the volume cannot be computed.

    # all triangles should have consistent orientation.
    # not necessary
    mesh.orient_triangles()
    o3d.visualization.draw_geometries([mesh])

    # 凸包计算 Mesh
    mesh.compute_vertex_normals()

    pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
    hull, _ = pcl.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    print(hull.get_volume() * 1e6)  # 864cm^2  0.000864M^2
    # 742.92
    # print(hull.get_oriented_bounding_box().volume())
    o3d.visualization.draw_geometries([pcl, hull_ls])
