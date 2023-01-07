# 基于 GPU 的碰撞检测算法

**周雨豪  软件92  2018013399**



## 1 实验环境

- **操作系统：**Windows 10
- **IDE：**Visual Studio 2022
- **GPU：**NVIDIA GeForce GTX 750 Ti
- **CUDA：**12.0
- **NVIDIA-SMI：**528.02
- **OpenGL：**4.6.0
- **SM 数量：**5
- **依赖的库：**CUDA Toolkit，FreeGLUT，<math.h>



## 2 项目结构

| 文件          | 描述 |
| ------------- | ---- |
| ball.hpp      |      |
| balllist.hpp  |      |
| board.hpp     |      |
| camera.hpp    |      |
| collision.cu  |      |
| collision.cuh |      |
| light.hpp     |      |
| main.cpp      |      |
| point.hpp     |      |



## 3 运行流程



## 4 演示方法





## 5 参考内容

- CUDA 环境
  - https://zhuanlan.zhihu.com/p/488518526
  - https://quasar.ugent.be/files/doc/cuda-msvc-compatibility.html
  - https://www.cnblogs.com/liaohuiqiang/p/9791365.html
  - https://zhuanlan.zhihu.com/p/64376059
  - https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
  - https://blog.csdn.net/weixin_54672021/article/details/119965884
- OpenGL
  - https://zhuanlan.zhihu.com/p/402397399
  - https://blog.csdn.net/m0_46821706/article/details/114597201
  - https://www.cnblogs.com/Fionaaa/p/15557163.html
- 碰撞检测算法
  - https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda