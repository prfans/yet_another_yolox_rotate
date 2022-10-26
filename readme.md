# yolox旋转目标检测2

 ## 简介

    支持simATM、ATSS、TOPK等正负样本匹配（可能存在一些问题，并没有详细调试，仅供借鉴）
    支持yolov5骨干网、yolov6骨干网（可能存在一些小问题）
    仅供借鉴！



## 训练

    修改exps\default里的配置文件后，直接python train.py



## 测试

    修改参数后，python demo.py



## onnx转换

    支持onnx方式部署：python export.py


​    