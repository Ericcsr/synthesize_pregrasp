# Synthesize Dexterous Nonprehensile Pregrasp for Ungraspable Objects

This is official implementation of paper: **Synthesize Dexterous Nonprehensile Pregrasp for Ungraspable Objects (SIGGRAPH 23)**

Authors: Sirui Chen (HKU, Stanford), Albert Wu(Stanford), C. Karen Liu(Stanford)


### Installation
Python Version: 3.9

```
pip install -r requirements.txt
``` 

Clone and install [Pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch) and [PytorchKinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics)

## Example of usage
In the case of the following environments, all intermediate files are prepared for you. You can start in reversed order and try each steps of our pipeline.

### Step 5. Visualize kinematics trajectory
This visualize the kinematics trajectory in Pybullet.

```
python solve_kin_trajectory.py --exp_name plate_20.0 --mode animate --env plate --add_physics --add_approach --save_name <name_to_save>
```
It will also generate a blender motion file in: `./data/blender` please install this [widget](https://github.com/huy-ha/pybullet-blender-recorder) in blender to load the motion file and render better images.

### Step 4. Solve IK
First solve keyframes
```
python solve_kin_trajectory.py --exp_name plate_20.0 --mode keypoints --env plate --add_physics --add_approach --has_floor --save_name <name_to_save>
```
After solving keyframes, you should see a visualization of each keyframes. If two consecutive frames are too different, please resolve keyframes.
After successfully solving keyframes, solve intermediate frames based on keyframes, noitice that the `name_to_save` here need to be the same as solving keyframes
```
python solve_kin_trajectory.py --exp_name plate_20.0 --mode interp --env plate --add_physics --add_approach --has_floor --save_name <name_to_save>
```

After solving intermediate frames, you should see the visualization of entire motion sequence. If the motion is too giggly or has interpenetration, please resolve intermediate frames.

### Step 3. Generate grasps
Generate grasps condition on final finger tip pose 
```
python neurals/scripts/generate_grasps.py --exp_name plate_20.0 --env plate
```
It will visualize 20 grasps generated by CVAE, please remember the id of the grasp you want and type the grasp ID you want to save at the end.

### Step 2. Trajectory optimization with physics
Optimize contact point and object trajectory with physics using MPPI
```
python model_optimize.py --exp_name plate --env plate --max_force 20 --name_score only_score_with_df --name_epoch 2980 --has_distance_field --validate
```
After optimization, you should see the contact points and object trajectory inside `data/videos/<exp_name>_<max_force>.gif`.

### Step 1. Ranking nodes on contact state graph
Generate reduced contact state graph based on score function
```
python neurals/scripts/generate_csg.py --has_distance_field --env plate
```

## Building your own demo
### TODO: make data preparation pipeline cleaner.

## Citation
If you find this project interesting and helpful, please consider citing our work as following.
```
@inproceedings{chen2022pregrasp,
 author = {Sirui Chen, Albert Wu, C. Karen Liu},
 booktitle = {{SIGGRAPH} '23: Special Interest Group on Computer Graphics and Interactive Techniques Conference},
 title = {Synthesize Dexterous Nonprehensile Pregrasp for Ungraspable Objects},
 year = {2023}
}
```