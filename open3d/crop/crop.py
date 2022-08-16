import open3d as o3d
import numpy as np

if __name__ == "__main__":
    print("Testing camera in open3d ...")

    print("Load a ply point cloud, crop it, and render it")
    # sample_ply_data = o3d.data.DemoCropPointCloud()
    pcd = o3d.io.read_point_cloud("fragment.ply")

    pcd_points = np.asarray(pcd.points)

    for x in np.nditer(pcd_points):
        print(x, end=", ")

    print(pcd)
    print(np.asarray(pcd.points))

    vol = o3d.visualization.read_selection_polygon_volume("cropped.json")
    chair = vol.crop_point_cloud(pcd)

    # print("->正在可视化点云")
    # o3d.visualization.draw_geometries([pcd])

    # Flip the pointclouds, otherwise they will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    chair.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # print("Displaying original pointcloud ...")
    # o3d.visualization.draw([pcd])
    # o3d.visualization.draw_geometries([pcd])

    print("Diplaying cropped pointcloud")
    # o3d.visualization.draw([chair])

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.02)
    o3d.visualization.draw([downpcd])
    # o3d.visualization.draw_geometries([downpcd])
    o3d.visualization.draw_geometries([downpcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
