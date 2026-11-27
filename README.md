# Per-Object DBL-GNG for Semantic Mapping

Repository providing the DBL-GNG source code for the paper (manuscript in preparation)
> Online Semantic Mapping of Indoor Environment with Topological Approach from Object Instance Segmentation<br/>
> Bharata Kukuh Hariyanto, Mohamad Yani, Moch. Iskandar Riansyah, Naoyuki Kubota, Fajar Hidayatullah and Muhammad Rio Fernanda

<p align="center">
  <img src="assets/pipeline.png" alt="Pipeline from RGB-D and LiDAR input to the semantic map and per-object DBL-GNG topological graphs" width="1150" />
</p>

*System overview. This repository provides the **DBL-GNG Topological Representation** block.*

This repository contains **only the DBL-GNG stage**: a ROS 2 node that accumulates the points of each mapped object and builds one topological graph per object with
[Distributed Batch Learning Growing Neural Gas](https://github.com/CornerSiow/DBL-GNG).
The segmentation, RGB-D point cloud, polygon and map fusion nodes of the pipeline are not included.

<p align="center">
  <img src="assets/overview.png" alt="Point cloud, polygon and DBL-GNG topological graph of mapped objects" width="900" />
</p>

*Left to right: object point clouds, 2D polygons, DBL-GNG topological graphs (red: nodes, green: edges).*

## Setup
Tested on ROS 2 Humble with Python 3 and NumPy.
```bash
mkdir -p ~/gng_ws/src
cp -r dbl_gng_objects ~/gng_ws/src/
cd ~/gng_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select dbl_gng_objects
source install/setup.bash
```

## Run the code
```bash
ros2 run dbl_gng_objects gng_objects_node
```
The node needs a semantic mapping system that publishes the two input topics below.
Input topic names can be changed with the parameters `points_topic`, `map_topic`, `reset_topic` and `clear_saved_topic`; use ROS remapping for the output topics.

| Direction | Topic | Type | Content |
|---|---|---|---|
| in | `/semantic_map/global_objects_pointcloud` | `sensor_msgs/PointCloud2` | fields `x`, `y`, `z`, `class_id` in the `map` frame |
| in | `/semantic_map/global_semantic_map` | `std_msgs/String` | JSON: `{"objects": [{"global_id", "class_id", "class_name", "bbox_map": {"min_bound", "max_bound"}, "centroid_map"}]}` (`class_name` is optional) |
| in | `/semantic_map/reset_map_cmd` | `std_msgs/Empty` | reset the graphs (archived if `persist_file` is set) |
| in | `/semantic_map/gng_clear_saved` | `std_msgs/Empty` | delete all graphs, including saved ones |
| out | `/semantic_map/gng_markers` | `visualization_msgs/MarkerArray` | nodes, edges and a class name label per object |
| out | `/semantic_map/gng_points` | `sensor_msgs/PointCloud2` | accumulated object points |

## Customize
| Parameter | Default | Meaning |
|---|---|---|
| `voxel` | 0.02 | voxel size (m) for point merging |
| `max_points` | 1500 | maximum points per object |
| `train_min_points` | 50 | minimum points before training |
| `train_interval_s` | 2.0 | minimum time between trainings (s) |
| `epochs` | 40 | training epochs |
| `margin` | 0.15 | bounding box margin (m) for assigning points to objects |
| `label_size` | 0.15 | text height (m) of the object name label, 0 hides the labels |
| `label_color` | [0.1, 0.1, 0.1] | label text color (RGB, 0 to 1); use `[1.0, 1.0, 1.0]` on a dark background |
| `persist_file` | empty | `.npz` file that keeps the graphs between sessions |

The number of nodes is `M = clip(N/15, 20, 200)` and the number of starting points is `K = clip(M/8, 3, 12)`, with `N` the number of points.

Example:
```bash
ros2 run dbl_gng_objects gng_objects_node --ros-args -p train_interval_s:=1.0 -p persist_file:=~/gng.npz
```

## Tests
```bash
cd dbl_gng_objects
python3 -m pytest
```

## Acknowledgement
`dbl_gng_core.py` is a port of [CornerSiow/DBL-GNG](https://github.com/CornerSiow/DBL-GNG) (MIT License, Mathematics 12(12):1909, 2024).
Please cite that work when using the algorithm.

## License
Apache-2.0, see [LICENSE](./LICENSE). The ported DBL-GNG core keeps the MIT License, see [LICENSE-DBL-GNG](./LICENSE-DBL-GNG).
