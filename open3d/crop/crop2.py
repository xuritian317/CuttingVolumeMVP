import open3d as o3d
import numpy as np
import json
import argparse


def draw_crop_box(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
        # print(data['bounding_polygon'])

    crop_point_pc = o3d.geometry.PointCloud()
    crop_point_pc.points = o3d.utility.Vector3dVector(data['bounding_polygon'])
    crop_point_pc.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([crop_point_pc])
    return crop_point_pc


def write_points_2file(arr):
    with open('frag_all_point.txt', 'a') as f:
        for x in arr:
            # print(x)
            f.write(str(x) + '\n')


if __name__ == "__main__":
    print("Load a ply point cloud, crop it, and render it")
    # sample_ply_data = o3d.data.DemoCropPointCloud()
    # pcd = o3d.io.read_point_cloud("fragment.ply")
    pcd = o3d.io.read_point_cloud("../cuttings2/19.ply")
    # o3d.visualization.draw_geometries([pcd])

    # pcd_points = np.asarray(pcd.points)
    # print(pcd)
    # arr = np.asarray(pcd.points)
    # write_points_2file(arr)

    crop_point = draw_crop_box('cropped_cutting2.json')

    vol = o3d.visualization.read_selection_polygon_volume("cropped_cutting2.json")
    tPcd = vol.crop_point_cloud(pcd)

    # print("->正在可视化点云")
    # o3d.visualization.draw_geometries([pcd])

    # print("Displaying original pointcloud ...")
    # o3d.visualization.draw([pcd])
    # o3d.visualization.draw_geometries([crop_point_pc])
    o3d.visualization.draw_geometries([pcd, crop_point])
    o3d.visualization.draw_geometries([tPcd, crop_point])

    downpcd = tPcd.voxel_down_sample(voxel_size=0.002)
    # 平面切割 获得多目标(立于平面之上)
    plane_model, inliers = downpcd.segment_plane(distance_threshold=0.0025,
                                                 ransac_n=5,
                                                 num_iterations=1000)

    # inlier_cloud = downpcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = downpcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw([outlier_cloud])

    # print("Diplaying cropped pointcloud")
    # o3d.visualization.draw([chair])

    # print("Downsample the point cloud with a voxel of 0.05")
    # downpcd = pcd.voxel_down_sample(voxel_size=0.02)
    # o3d.visualization.draw([downpcd])
    # # o3d.visualization.draw_geometries([downpcd])
    # o3d.visualization.draw_geometries([downpcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])
