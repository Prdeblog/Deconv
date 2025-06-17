# DEConv-YOLO

This is an official PyTorch implementation for the paper **DEConv-YOLO: An approach based on multi-scale feature fusion with lightweight convolutional design for tomato leaf disease detection**.

## Core Features

- **Efficient Lightweight Design**: Drawing inspiration from depthwise separable convolution (DSWConv) and the DEA-Net module, we redesigned the C2f layer and backbone network to form a **C2f-DEConv** module. By integrating various differential convolutions (CDC, ADC, VDC, HDC) to replace certain standard convolutional layers, the model reduces computational complexity and parameter counts.
- **Enhanced Multi-scale Feature Extraction**:
  - The Adown module was introduced to strengthen the detection of small objects like tiny lesions.
  - It combines multiple Concat operations and the SPPF module to facilitate cross-layer information sharing and multi-scale feature fusion, thereby improving feature propagation efficiency.
- **Precise Object Localization**: **Shape-IoU Loss** was adopted as the bounding box regression loss function, which improves localization accuracy by considering the inherent shape and scale attributes of bounding boxes.

## Performance

We conducted comprehensive experiments on the **PlantVillage** dataset and compared DEConv-YOLO with current state-of-the-art models such as YOLOv8, YOLOv9, and RT-DETR.

**Key Results:**

**Performance Surpassed**: Compared to the high-performing YOLOv8x baseline model, DEConv-YOLO shows improvements across several key metrics:

**mAP@50-95**: Increased by **2.9%** (from 60.2% to 63.1%) 

**mAP@50**: Increased by **2.9%**

**Accuracy**: Increased by **1.5%**

**Recall**: Increased by **1.7%**

**F1-score**: Increased by **2.3%**

**Model Lightweighting**: While achieving performance gains, the model's parameter count was reduced by **12.4%**.

**Robustness**: The model also demonstrates strong robustness on datasets simulating various adverse environments (e.g., changes in brightness, blur, compression).

### Data

This study utilized the public **Plant Village** dataset. We also constructed an extended dataset called **Plant Village-expanded** by using data augmentation techniques such as adding noise and blur to improve the model's robustness and generalization in real-world complex scenarios.Link to data:https://github.com/mamta-joshi-gehlot/Tomato-Village

### Test

Under run/train, we provide our specific weights and logs, as well as visualization of the training process. You can call the weight implementation test.
Execute

```
 python test.py
```

### Train

If you want to train, please download the above dataset, unzip it to a fixed location, and replace your path in data.ymal.

Execute the code 

```
python train.py
```

