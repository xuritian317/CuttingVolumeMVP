import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


if __name__ == "__main__":
    print("Load a ply point cloud, print it, and render it")

    print("->正在加载点云... ")
    pcd = o3d.io.read_point_cloud("crop/fragment.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    # o3d.visualization.draw_geometries([downpcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    print("Load a polygon volume and use it to crop the original point cloud")
    vol = o3d.visualization.read_selection_polygon_volume("crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    # o3d.visualization.draw_geometries([chair],
    #                                   zoom=0.7,
    #                                   front=[0.5439, -0.2333, -0.8060],
    #                                   lookat=[2.4615, 2.1331, 1.338],
    #                                   up=[-0.1781, -0.9708, 0.1608])

    # print("Paint chair")
    # chair.paint_uniform_color([1, 0.706, 0])
    # o3d.visualization.draw_geometries([chair],
    #                                   zoom=0.7,
    #                                   front=[0.5439, -0.2333, -0.8060],
    #                                   lookat=[2.4615, 2.1331, 1.338],
    #                                   up=[-0.1781, -0.9708, 0.1608])

    # 点云差值
    dists = pcd.compute_point_cloud_distance(chair)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.01)[0]
    pcd_without_chair = pcd.select_by_index(ind)
    # o3d.visualization.draw_geometries([pcd_without_chair],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    # 最小凸包
    mesh = o3d.io.read_triangle_mesh("BunnyMesh.ply")
    mesh.compute_vertex_normals()

    pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
    hull, _ = pcl.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    print(hull.get_volume())  # It should be 1 but it gives me 0
    print(hull.get_oriented_bounding_box().volume())  # As a control (returns 1).
    o3d.visualization.draw_geometries([pcl, hull_ls])

    # 聚类算法
    pcd = o3d.io.read_point_cloud("crop/fragment.ply")

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=0.455,
    #                                   front=[-0.4999, -0.1659, -0.8499],
    #                                   lookat=[2.1813, 2.0619, 2.0999],
    #                                   up=[0.1204, -0.9852, 0.1215])

    # Plane segmentation
    pcd = o3d.io.read_point_cloud("fragment.pcd")
    # o3d.visualization.draw(pcd)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
    #                                   zoom=0.8,
    #                                   front=[-0.4999, -0.1659, -0.8499],
    #                                   lookat=[2.1813, 2.0619, 2.0999],
    #                                   up=[0.1204, -0.9852, 0.1215])

    # Hidden point removal
    print("Convert mesh to a point cloud and estimate dimensions")
    # armadillo = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh("ArmadilloMesh.ply")
    # o3d.visualization.draw(mesh)
    # o3d.visualization.draw_geometries([mesh])
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_poisson_disk(5000)
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    # o3d.visualization.draw_geometries([pcd])

    print("Define parameters used for hidden_point_removal")
    camera = [0, 0, diameter]
    radius = diameter * 100

    print("Get all points that are visible from given view point")
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    print("Visualize result")
    pcd = pcd.select_by_index(pt_map)
    # o3d.visualization.draw_geometries([pcd])

    # Point cloud outlier removal
    pcd = o3d.io.read_point_cloud("fragment.pcd")
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    # o3d.visualization.draw_geometries([voxel_down_pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    print("Every 5th points are selected")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    # o3d.visualization.draw_geometries([uni_down_pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    print("Statistical oulier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
    # display_inlier_outlier(voxel_down_pcd, ind)

    print("Radius oulier removal")
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    # display_inlier_outlier(voxel_down_pcd, ind)

    # # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # o3d.io.write_point_cloud("../../test_data/sync.ply", pcd)
    # # Load saved point cloud and visualize it
    # pcd_load = o3d.io.read_point_cloud("../../test_data/sync.ply")
    #
    # # Convert Open3D.o3d.geometry.PointCloud to numpy array
    # xyz_load = np.asarray(pcd_load.points)
    # print('xyz_load')
    # print(xyz_load)
    # o3d.visualization.draw_geometries([pcd_load])

    # print("->正在保存点云")
    # o3d.io.write_point_cloud("write.pcd", pcd, True) # 默认false，保存为Binarty；True 保存为ASICC形式
    # print(pcd)

    # o3d.t.io.RealSenseSensor.list_devices()

    # bag_reader = o3d.t.io.RSBagReader()
    # bag_reader.open("L515_test.bag")
    # im_rgbd = bag_reader.next_frame()
    # while not bag_reader.is_eof():
    #     # process im_rgbd.depth and im_rgbd.color
    #     im_rgbd = bag_reader.next_frame()
    #
    # bag_reader.close()

    # import json
    #
    # with open("camera_setting.json") as cf:
    #     rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))
    #
    # rs = o3d.t.io.RealSenseSensor()
    # rs.init_sensor(rs_cfg, 0, "L515_new_get.bag")
    # rs.start_capture(True)  # true: start recording with capture
    # for fid in range(150):
    #     im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
    #     # process im_rgbd.depth and im_rgbd.color
    #
    # rs.stop_capture()
