# Continual Learning for Traversability Prediction with Uncertainty-Aware Adaptation
<p align="center">
  <img src="https://github.com/user-attachments/assets/19b7bb4d-8421-4834-a76e-d6a4b97a6918" alt="fig1" width="600">
</p>

--------
  TBD

```bibtex
@article{Lee2025ContinualTraversability,
  author  = {Lee, Hojin and others},
  title   = {Continual Learning for Traversability Prediction with Uncertainty-Aware Adaptation},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2025}
}
```
Overview
--------
This repository implements:
- A continual learning pipeline for traversability prediction that adapts across environments.
- An off-road navigation stack that utilizes the learned traversability model.
- Terrain Feature recording and processing nodes.

## System Setup

- 🖥️ **Hardware**: NVIDIA Jetson AGX Orin Developer Kit  
- 📦 **JetPack**: 6.1 — L4T 36.4.0 — CUDA 12.6  
- 🔬 **Frameworks**:  
  - PyTorch 2.5.0  
  - JAX 0.4.35  
- 🤖 **ROS 2**: Humble Hawksbill (CycloneDDS)


## Packages & Launch Files
| Package          | File                                   | Description                                                    |
|------------------|----------------------------------------|----------------------------------------------------------------|
| `align_planner`  | `align_planner/data_logger.launch.py`  | Record traversability parameters for training.                 |
| `align_planner`  | `align_planner/train/mcmc_train_main.py` | Continual-learning training script for the traversability model. |
| `align_planner`  | `align_planner/align_planner.launch.py`| Run off-road navigation with the learned model.                |
| `feat_processing`| `feat_processing/feat_recording.launch.py` | Record DINOv2 terrain features for the encoder.               |
| `feat_processing`| `feat_processing/feat_processing.launch.py`| Run the trained feature encoder online.                        |


  
## Training Data Layout

| Type                          | Path                                                                                                      | Notes                                                                 |
|-------------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| Traversability model training | `/home/user/align_ws/install/align_planner/share/align_planner/`<br>`<env_data_folder>/<sub_envs_data_folder>/`<br>`grid_state_data_<timestamp>.pt` | - `<env_data_folder>` = environment name (e.g., `park`)<br>- `<sub_envs_data_folder>` = session ID<br>- `grid_state_data_<  >.pt` = training data |
| Feature encoder training data | `/home/user/align_ws/install/feat_processing/share/feat_processing/data/`<br>`<env_datafolder>/buffer_<timestamp>.pt` | - `<env_datafolder>` = environment name<br>- `buffer_<timestamp>.pt` = raw feature buffer for encoder training |



Continual Learning Setup
------------------------
Edit the environment curriculum list in your training script:

  In align_planner/train/mcmc_train_main.py:
    Tasks_dirs = ["park_summer", "bike_trail", "sand", "river", "forest_summer"]

Each entry corresponds to one environment (task) in the continual learning sequence.
Order reflects the training curriculum (earlier to later tasks).

## Feature → Elevation Mapping Integration

The `feat_processing` node publishes a **multi-dimensional terrain feature vector**, which is consumed by the **elevation mapping** node.

- **Upstream (features):** this repo → `feat_processing`  
- **Downstream (fusion/mapping):** [`leggedrobotics/elevation_mapping_cupy`](https://github.com/leggedrobotics/elevation_mapping_cupy.git)  

** ROS 2 Elevation Mapping Version **

For ROS 2 usage in this project, we rely on the following fork:
👉 [amilearning/elevation_mapping_cupy_ros2](https://github.com/amilearning/elevation_mapping_cupy_ros2/tree/orin)

---
