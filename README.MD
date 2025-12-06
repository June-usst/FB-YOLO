###This document explains the code structure of FB-YOLO.###
In ultralytics/nn/modules/block.py, we define the SE-DCCS module, which is the DCSSE mentioned in our article.
In ultralytics/nn/modules/__init__.py, we import the enhanced modules and classes.
The function for adding various enhancement modules to the module is defined in ultralytics/nn/modules/task.py.
The definition of the AS-EI-OU loss function has been added to ultralytics/utils/metrics.py. Within the module, we refer to it as eneiou.
The feature extraction framework of pplcnet is directly imported from external libraries and is not defined within this framework.
Under the YAML files, yolov8_lcnet_075.yaml utilizes only the LCnet feature extraction framework; yolov8-se_dccsp23.yaml employs only the DCSSE module as an enhancement; yolov8_lcnet_sedccs.yaml represents our final FB-YOLO framework.
In 1_yolov8_train.py, you can opt to use the as-eiou function (eneiou).
The run/detect folder contains numerous run results, which can be ignored.
AirbirdsData and ChongmingData are both pre-configured datasets. Their paths can be set in data.yaml for direct use.