import tensorflow as tf
import h5py
import os
import fnmatch
import cv2
import numpy as np
from tqdm import tqdm

def decode_img(img):
    """Decode an image from bytes to RGB format."""
    return cv2.cvtColor(cv2.apend(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def _bytes_feature(value):
    """Convert a value to a bytes_list feature."""
    if isinstance(value, type(tf.constant(0))):  # Convert TensorFlow tensor to numpy
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bool_feature(value):
    """Convert a boolean to an int64_list feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def serialize_example(action, qpos, cam_high, cam_left_wrist, cam_right_wrist, terminate_episode):
    """Serialize an example to a TFRecord-compatible format."""
    feature = {
        'action': _bytes_feature(tf.io.serialize_tensor(action)),
        'qpos': _bytes_feature(tf.io.serialize_tensor(qpos)),
        'cam_high': _bytes_feature(tf.io.serialize_tensor(cam_high)),
        'cam_left_wrist': _bytes_feature(tf.io.serialize_tensor(cam_left_wrist)),
        'cam_right_wrist': _bytes_feature(tf.io.serialize_tensor(cam_right_wrist)),
        #'instruction': _bytes_feature(instruction),
        'terminate_episode': _bool_feature(terminate_episode)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def check_hdf5_file(filepath):
    """Check if an HDF5 file contains only required datasets."""
    required_datasets = [
        'action',
        'observations/qpos',
        'observations/images/cam_high',
        'observations/images/cam_left_wrist',
        'observations/images/cam_right_wrist',
        #'instruction'
    ]
    try:
        with h5py.File(filepath, 'r') as f:
            for dataset in required_datasets:
                if dataset not in f:
                    print(f"Missing dataset '{dataset}' in file: {filepath}")
                    return False
            return True
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return False

def write_tfrecords(root_dir, out_dir):
    """Convert HDF5 files to TFRecords while removing unnecessary datasets."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_files = 0
    for root, dirs, files in os.walk(root_dir):
        num_files += len(fnmatch.filter(files, '*.hdf5'))
    print(f"Total HDF5 files found: {num_files}")

    with tqdm(total=num_files) as pbar:
        for root, dirs, files in os.walk(root_dir):
            for filename in fnmatch.filter(files, '*.hdf5'):
                filepath = os.path.join(root, filename)

                # Check if the HDF5 file meets the required format
                if not check_hdf5_file(filepath):
                    print(f"Skipping invalid file: {filepath}")
                    pbar.update(1)
                    continue

                try:
                    with h5py.File(filepath, 'r') as f:
                        pbar.update(1)

                        output_dir = os.path.join(out_dir, os.path.relpath(root, root_dir))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        print(f"Writing TFRecords to {output_dir}")
                        tfrecord_path = os.path.join(output_dir, filename.replace('.hdf5', '.tfrecord'))

                        with tf.io.TFRecordWriter(tfrecord_path) as writer:
                            num_episodes = f['action'].shape[0]

                            for i in range(num_episodes):
                                # Extract required data
                                action = f['action'][i]
                                qpos = f['observations']['qpos'][i]
                                cam_high = decode_img(f['observations']['images']['cam_high'][i])
                                cam_left_wrist = decode_img(f['observations']['images']['cam_left_wrist'][i])
                                cam_right_wrist = decode_img(f['observations']['images']['cam_right_wrist'][i])
                                #instruction = f['instruction'][()]
                                terminate_episode = i == num_episodes - 1

                                # Serialize and write
                                serialized_example = serialize_example(action, qpos, cam_high, cam_left_wrist, cam_right_wrist, terminate_episode)
                                writer.write(serialized_example)

                        print(f"TFRecords written to {tfrecord_path}")

                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")

    print(f"TFRecords written to {out_dir}")

# Define input and output directories
root_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/data/aloha_mobile_dummy/aloha_mobile_dummy/'
output_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/data/aloha_mobile_dummy/tf/'

# Convert HDF5 files to TFRecords while filtering out unnecessary datasets
write_tfrecords(root_dir, output_dir)
