脚本的目的是将 .hdf5 文件中的多种数据格式转换成 TFRecord 格式，便于 TensorFlow 进行高效训练。
主要步骤: 读取 .hdf5 文件 → 提取数据 → 序列化数据 → 将序列化的数据写入 .tfrecord 文件。
使用的工具: TensorFlow 用于序列化数据和写入 TFRecord，h5py 用于读取 .hdf5 文件，cv2 用于解码图像。