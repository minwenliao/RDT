import h5py
import os
import argparse

def is_valid_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            # Attempt to read all keys
            keys = list(f.keys())
            # If reading keys succeeds without exceptions, the file is likely intact
            return True
    except Exception as e:
        print(f"Corrupted file {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Validate HDF5 files')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the data')
    parser.add_argument('--output', type=str, default='valid_hdf5.txt', help='Path to the output file')
    args = parser.parse_args()

    valid_files = []
    for root, _, files in os.walk(args.data_root):
        for file in files:
            if file.endswith('.hdf5'):
                full_path = os.path.join(root, file)
                if is_valid_hdf5(full_path):
                    valid_files.append(full_path)

    with open(args.output, 'w') as f:
        f.write('\n'.join(valid_files))

    print(f"Generated list of valid files: {args.output}, total {len(valid_files)} valid files")

if __name__ == '__main__':
    main()