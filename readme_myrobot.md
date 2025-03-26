在新数据集（例如 my_pretrain_dataset）上进行训练，需要修改以下几个文件：
1. configs/dataset_control_freq.json
添加您数据集的控制频率。

2. data/preprocess_scripts/my_pretrain_dataset.py
如果您的数据集可以通过 tfds.builder_from_directory() 加载，则只需将其下载到 Open X-Embodiment 数据目录 data/datasets/openx_embod，并实现 process_step() 函数。您可能需要在第78行指定 tfds 加载路径（请参考该文件）。我们参考了 data/preprocess_scripts/droid.py 作为示例。

如果不行，您需要将数据集转换为 TFRecords 格式，然后分别实现 load_dataset() 和 process_step()。我们参考了 data/agilex/hdf5totfrecords.py 和 data/preprocess_scripts/agilex.py 作为示例。

3. configs/dataset_img_keys.json
添加您数据集的图像键。例如：

json
复制
编辑
"my_pretrain_dataset": {
  "image_keys": [
    "exterior-cam",
    "right-wrist-cam",
    "left-wrist-cam",
    "left-wrist-cam"
  ],
  "image_mask": [1, 1, 1, 0]
}
4. configs/dataset_stat.json
计算数据集的统计信息（最小值、最大值、均值和标准差）：

bash
复制
编辑
# 使用 -h 查看完整用法
python -m data.compute_dataset_stat --skip_exist
5. data/vla_dataset.py
如果您的数据集不能通过 tfds.builder_from_directory() 加载，则将其添加到 DATASET_NAMES_NOOPENX 中。
如果您的数据集只包含动作而没有自我状态（即机器人状态），则将其添加到 DATASET_NAMES_NO_STATE 中（请参考该文件）。
如果您希望使用不同的动作，请实现更多函数，参考 flatten_episode_agilex() 和 _generate_json_state_agilex()。


python -m data.producer --fill_up
# 填充过程完成后，继续下一步
然后运行预训练脚本：

bash
复制
编辑
source pretrain.sh


1.
/home/agilex/RoboticsDiffusionTransformer-main/scripts/encode_lang.py

#after make my TASK_NAME and INSTRUCTION 

# python -m scripts.encode_lang
生成语言嵌入文件


2.
生成语言嵌入文件后，你可以在 inference.sh 或推理命令中指定该文件的路径：

python -m scripts.agilex_inference \
    --use_actions_interpolation \
    --pretrained_model_name_or_path="checkpoints/your_finetuned_ckpt.pt" \  # your finetuned checkpoint: e.g., checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>, checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt,
    --lang_embeddings_path="outs/lang_embeddings/your_instr.pt" \
    --ctrl_freq=25    # your control frequency

--lang_embeddings_path=outs/lang_embeddings/your_instr.pt


google/siglip-so400m-patch14-384



roscore
