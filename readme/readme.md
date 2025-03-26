# Fine-tune RoboticsDiffusionTransformer 过程整理

本文档整理了从数据采集、预处理、语言指令编码、HDF5 数据解析，到数据集链接与环境配置的整个流程。

---

## 1. 数据采集与处理

### 1.1 采集数据

- 使用 `collectdata.py` 进行数据采集，生成 **HDF5** 格式的数据文件。

### 1.2 过滤 HDF5 数据

- 运行以下脚本，删除 HDF5 文件中不必要的信息，仅保留需要的数据：

  ```bash
  python /baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/data/get_expected_format_/delet_hdf5_not_nessecy.py

### 1.3 HDF5 转 TFRecords(要语言嵌入的时候采用)
-  执行以下脚本，将 HDF5 格式的数据转换为 TFRecords 格式，以适用于后续训练流程：

    ```bash
    python /baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-ma

- 生成 .pt 语言嵌入文件
运行以下脚本，对语言指令进行嵌入编码，生成 .pt 文件：

    ```bash
    python /baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/scripts/encode_lang_batch.py
    注意：请记录生成的 .pt 文件路径，后续需要在 hdf5_vla_dataset.py 中配置该路径。

### 当卡