finetune
首先用collectdata.py进行数据采集生成hdf5格式的文件

接下来用/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/data/get_expected_format_/delet_hdf5_not_nessecy.py
只要hdf5格式需要读入的东西

用/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/data/get_expected_format_/hdf5_to_tr.py将hdf5转为tfrecords格式

/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/scripts/encode_lang_batch.py对语言进行嵌入编码
生成.pt文件 记下这个路径 放入/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/data/hdf5_vla_dataset.py中hdf5类的instruction路径

其中的
json文件包换gpt扩充指令 原指令 简化指令
这里我取上面三者的比例为100:1:1

接下来用
/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/data/hdf5_vla_dataset.py
和
/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/data/compute_dataset_stat_hdf5.python
对hdf5进行解析，解析完会在code/2_3/2/RoboticsDiffusionTransformer-main/configs/dataset_stat.json最底下有解析完的数据

在三个json文件
configs/dataset_control_freq.json
configs/finetune_datasets.json
configs/finetune_sample_weights.json
加入数据集的名字

链接数据集


source finetune.sh

下载pytorch的时候
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://mirrors.ustc.edu.cn/pypi/web/simple

在下载flash-attn的时候遇到了cuda多版本问题 
尝试：忘了

ln -s /baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/rdt/siglip-so400m-patch14-384 /baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/google

