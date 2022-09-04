import open3d as o3d
import numpy as np
import json

if __name__ == "__main__":
    print("Load a ply point cloud, crop it, and render it")
    # sample_ply_data = o3d.data.DemoCropPointCloud()
    pcd = o3d.io.read_point_cloud("fragment.ply")

    # pcd_points = np.asarray(pcd.points)

    # print(pcd)
    # arr = np.asarray(pcd.points)
    # print(type(arr))
    # print(arr)

    with open('cropped.json', 'r') as f:
        data = json.load(f)
        print(data['bounding_polygon'])
        pass

    crop_point_pc = o3d.geometry.PointCloud()
    crop_point_pc.points = o3d.utility.Vector3dVector(data['bounding_polygon'][:2])
    crop_point_pc.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([crop_point_pc])

    crop_point_pc1 = o3d.geometry.PointCloud()
    crop_point_pc1.points = o3d.utility.Vector3dVector(data['bounding_polygon'][2:4])
    crop_point_pc1.paint_uniform_color([1, 1, 0])

    crop_point_pc2 = o3d.geometry.PointCloud()
    crop_point_pc2.points = o3d.utility.Vector3dVector(data['bounding_polygon'][4:6])
    crop_point_pc2.paint_uniform_color([1, 0, 1])

    crop_point_pc3 = o3d.geometry.PointCloud()
    crop_point_pc3.points = o3d.utility.Vector3dVector(data['bounding_polygon'][6:8])
    crop_point_pc3.paint_uniform_color([0, 0, 0])

    # with open('frag_all_point.txt', 'a') as f:
    #     for x in arr:
    #         # print(x)
    #         f.write(str(x) + '\n')

    vol = o3d.visualization.read_selection_polygon_volume("cropped.json")
    chair = vol.crop_point_cloud(pcd)

    # print("->正在可视化点云")
    # o3d.visualization.draw_geometries([pcd])

    # Flip the pointclouds, otherwise they will be upside down.
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # chair.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # crop_point_pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # print("Displaying original pointcloud ...")
    # o3d.visualization.draw([pcd])
    # o3d.visualization.draw_geometries([crop_point_pc])
    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([chair])
    o3d.visualization.draw_geometries([pcd, crop_point_pc, crop_point_pc1, crop_point_pc2, crop_point_pc3])

    print("Diplaying cropped pointcloud")
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
