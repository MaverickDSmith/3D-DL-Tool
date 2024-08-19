from docarray import DocList
import open3d as o3d
import numpy as np
import os


from models.doc_structure import PointCloud




##TODO:
# 1.) Generalize this to most representations of Objects/Meshes/Pointclouds
# Such as:
# .obj
# .ply
# .stl
# etc...
# try to get everything Open3D supports at least
#
# 2.) Multi-Label Support
# meaning, if a folder is within a folder such as
# root
# - label1
# - - label 1_1
# - - - object
# we can support multiple labels
# example
# root : truck : ford : ford.obj, | root : truck : chevy : chevy.obj
# could use this as an in for multi-label training for further supervised learning tasks
# but remember this initial implementation is meant to be pure pointcloud classification 
# and robust feature learning methods

def downsample(points):
    

    return points

def create_pointcloud_docs(root_dir):
    docs = DocList[PointCloud]()
    label_counter = 0
    
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        
        # Ensure we are only processing directories
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.off'):
                    off_path = os.path.join(dir_path, file_name)
                    
                    # Create the PointCloud object
                    points = o3d.io.read_triangle_mesh(off_path)
                    points = downsample(points)
                    pointcloud_doc = PointCloud(
                        points=points,
                        embedding=None,
                        label=label_counter,
                        object=dir_name
                    )
                    
                    # Append to the list
                    docs.append(pointcloud_doc)
                    
                    # Increment the label counter
                    label_counter += 1
    
    return docs