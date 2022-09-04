import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

config_parser = parser = argparse.ArgumentParser(description='Cutting Volume Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('--data_dir', metavar='DIR', default='', type=str,
                    help='path to dataset')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def test_args():
    args, args_text = _parse_args()
    print('test_args')


def main():
    #
    print("try to calculate the volume of cutting ... ")
    print("->正在加载点云... ")
    pcd = o3d.io.read_point_cloud("../cuttings2/18.ply")
    # print(pcd)
    # print(np.asarray(pcd.points))

    # print("->正在可视化点云")
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw([pcd])

    # 点云裁切
    vol = o3d.visualization.read_selection_polygon_volume("cropped_cutting2.json")
    crop_pcd = vol.crop_point_cloud(pcd)
    # o3d.visualization.draw([crop_pcd])

    # 点云下采样
    # downpcd = crop_pcd.uniform_down_sample(every_k_points=3)
    downpcd = crop_pcd.voxel_down_sample(voxel_size=0.002)
    # downpcd = crop_pcd
    # o3d.visualization.draw([pcd])
    o3d.visualization.draw_geometries([downpcd])
    o3d.io.write_point_cloud('d_5.pcd', downpcd)

    # 平面切割 获得多目标(立于平面之上)
    plane_model, inliers = downpcd.segment_plane(distance_threshold=0.0025,
                                                 ransac_n=5,
                                                 num_iterations=1000)

    # inlier_cloud = downpcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = downpcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([outlier_cloud])

    # 离群值删除
    cl, ind = outlier_cloud.remove_radius_outlier(nb_points=30, radius=0.02)
    target_cloud = display_inlier_outlier(outlier_cloud, ind)

    # 同类聚集
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(target_cloud.cluster_dbscan(eps=0.01, min_points=25, print_progress=False))

    # for x in labels:
    #     print(x)

    # print(labels)

    # labels = labels[labels > 0]
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    target_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([target_cloud])
    # print(colors)

    # 根据聚集结构构造主要目标
    ori_pc = np.asarray(target_cloud.points)
    print(ori_pc.size)
    print(labels.size)

    labels_unique = np.unique(labels)
    labels_unique = labels_unique[labels_unique >= 0]
    print(labels_unique)

    # pcs = []  # 所有岩屑堆
    count = 0
    volumes = 0
    for index in labels_unique:
        pc = []
        for label, point in zip(labels, ori_pc):
            if label == index:
                pc.append(point)
        print(len(pc))

        volumes += cal_pc_volumes(pc)
        # volumes += cal_pc_volume_mesh(pc)
        count += len(pc)

    print(volumes)


def cal_pc_volumes(pc):
    obj = o3d.geometry.PointCloud()
    obj.points = o3d.utility.Vector3dVector(pc)
    # o3d.visualization.draw_geometries([obj])

    hull, _ = obj.compute_convex_hull()  # [hull]:
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    # o3d.visualization.draw_geometries([hull_ls])

    volume = hull.get_volume() * 1e6
    print(f"[hull]: {volume}")  # 110 CM^2
    return volume


def cal_pc_volume_mesh(pc):
    # pointcloud 转 mesh
    obj = o3d.geometry.PointCloud()
    obj.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([obj])

    # obj.estimate_normals()
    radius = 0.1  # 搜索半径
    max_nn = 30  # 邻域内用于估算法线的最大点数
    obj.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 法线估计

    # estimate radius for rolling ball
    # distances = obj.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist

    radii = [0.005, 0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        obj,
        o3d.utility.DoubleVector(radii))

    # obj.paint_uniform_color([1.0, 0.0, 1.0])
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(obj, depth=5)
    # print(mesh)

    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh])

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    #     obj,
    #     alpha=0.1,
    # )
    # o3d.visualization.draw_geometries([mesh])
    # print(mesh.get_volume() * 1e6)  # The mesh is not watertight, and the volume cannot be computed.

    # all triangles should have consistent orientation.
    # not necessary

    # mesh.orient_triangles()
    # o3d.visualization.draw_geometries([mesh])

    print(mesh.is_watertight())

    # 凸包计算 Mesh
    # mesh.compute_vertex_normals()

    pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
    hull, _ = pcl.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    # print(hull.get_volume() * 1e6)
    # print(hull.get_oriented_bounding_box().volume())
    # o3d.visualization.draw_geometries([pcl, hull_ls])
    volume = hull.get_volume() * 1e6
    # volume = hull.get_oriented_bounding_box().volume() * 1e6
    print(f"[hull]: {volume}")  # 110 CM^2
    return volume


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud


if __name__ == "__main__":
    main()
    # test_args()
