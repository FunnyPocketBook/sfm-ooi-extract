import open3d as o3d
import numpy as np

def flip_ply_on_z_axis(input_file, output_file):
    # Read the .ply file
    point_cloud = o3d.io.read_point_cloud(input_file)
    if not point_cloud.is_empty():
        print(f"Loaded point cloud with {len(point_cloud.points)} points.")

        # Flip on Z-axis by negating the Z-coordinates
        points = np.asarray(point_cloud.points)
        points[:, 1] = -points[:, 1]
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # Save the flipped point cloud to a new .ply file
        o3d.io.write_point_cloud(output_file, point_cloud)
        print(f"Flipped point cloud saved to {output_file}.")
    else:
        print(f"Error: The point cloud is empty. Check the input file: {input_file}")

# Input and output file paths
input_ply_file = r"S:\git\sfm-ooi-extract\output\kitchen\exp1\kitchen_floorseg.ply"  # Replace with the path to your .ply file
output_ply_file = r"S:\git\sfm-ooi-extract\output\kitchen\exp1\kitchen_floorseg_flipped.ply"  # Replace with the desired output file path

# Call the function
flip_ply_on_z_axis(input_ply_file, output_ply_file)
