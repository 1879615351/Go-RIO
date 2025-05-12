<p align="center">
  <h1 align="center">Go-RIO: Ground-Optimized 4D Radar-Inertial Odometry via Continuous Velocity Integration using Gaussian Process</h1>
  <p align="center"><strong>[ICRA 2025 Best Paper Award Finalist]</strong></p>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=lh2KUKMAAAAJ"><strong>Wooseong Yang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=liSzSegAAAAJ"><strong>Hyesu Jang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=7yveufgAAAAJ"><strong>Ayoung Kim</strong></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2502.08093">Arxiv</a> | <a href="https://www.youtube.com/watch?v=0FnJ_BZe3vo&t=9s">Video</a></h3>
  <div align="center"></div>
</p>

## News
* **[24/04/2025]**: Our paper is selected as the **finalist** for ICRA 2025 **Best Paper Award**!
* **[29/01/2025]**: Our paper is accepted to ICRA 2025.

## Run
Full Source Code will be available soon! (Almost there!)

We recommend using the provided Docker environment (Ubuntu 20.04) for testing our code.

1. Clone our repository
    ```bash
    mkdir -p gorio_ws/src && cd gorio_ws/src
    git clone https://github.com/wooseongY/Go-RIO .
    ```
2. Build the Docker image
   ```bash
   cd docker
   sudo chmod +x build.sh
   ./build.sh
   ```
3. Modify the DATA_DIR in the run.sh as the appropriate data folder
4. Run the Docker container and build the package
    ```bash
    sudo chmod +x run.sh
    ./run.sh
    catkin_make && source devel/setup.bash
    ```
5. Modify the bag file path in the rosbag_play_\<sequence\>.launch file as "/root/data/<your_bag_directory>", which contains your proper bag file.
6. Launch our algorithm and enjoy :)
   ```bash
   roslaunch gorio <launch file name>.launch
   rostopic pub /command std_msgs/String "output_aftmapped"
   ```

## Citation
If you use our paper for any academic work, please cite our paper.
```bibtex
@INPROCEEDINGS {wsyang-2025-icra,
    author={Wooseong Yang and Hyesu Jang and Ayoung Kim},
    title={Ground-Optimized 4D Radar-Inertial Odometry via Continuous Velocity Integration using Gaussian Process},
    booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
    year={2025},
    month={May.},
    address={Atlanta},
}
```

## Contact
If you have any questions, please contact:
- Wooseong Yang ([wseongy15@gmail.com]())

## Acknowledgement 
This work was supported by the Robotics and AI (RAI) Institute, and in part by the MOITE, Korea (No. 1415187329).

And thanks for authors of [UGPM](https://github.com/UTS-RI/ugpm), [4DRadarSLAM](https://github.com/zhuge2333/4DRadarSLAM), [Patchwork](https://github.com/url-kaist/patchwork-plusplus-ros) and [REVE](https://github.com/christopherdoer/reve).


