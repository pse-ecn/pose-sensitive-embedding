import argparse
import glob
import ntpath
import os
import shutil
import sys

import cv2
import numpy as np
from scipy.misc import imread, imresize

#print(sys.path.append(os.path.dirname('images_to_pose_maps.py') + "/../"))

from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input
from demo.utils import pose_map_to_image, heat_map_to_image

TARGET_HEIGHT = 512


def calculate_and_write_pose_maps(image_paths, output_directory, mode):
    cfg = load_config("demo/pose_cfg.yaml")

    # Load and setup CNN part detector
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    for i, path in enumerate(image_paths):
        sys.stdout.write('\r >> Evaluating path %d of %d' % (i + 1, len(image_paths)))
        sys.stdout.flush()

        # Read image from file
        image = imread(path, mode='RGB')

        target_width = image.shape[1] * TARGET_HEIGHT / image.shape[0]

        scaled_image = imresize(image, (int(TARGET_HEIGHT), int(target_width)), 'cubic')
        image_batch = data_to_input(scaled_image)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref,_ = predict.extract_cnn_output(outputs_np, cfg)

        write_scmap(output_directory, ntpath.basename(path), scmap, mode, image.shape)

    print('\nFinished generating pose maps...')


def write_scmap(output_directory, file_name, scmap, mode, original_image_shape):
    file_type = '.png' if mode is not 'np' else '.npy'

    if mode == 'png':
        write_combined_pose_map(file_name, file_type, original_image_shape, output_directory, scmap)
            elif mode == 'channel-pngs':
        write_channel_pose_maps(file_name, file_type, original_image_shape, output_directory, scmap)
        write_combined_pose_map(file_name, file_type, original_image_shape, output_directory, scmap)

    elif mode == 'np':
        write_numpy_file(file_name, file_type, output_directory, scmap)

    else:
        raise ValueError('Unknown mode %s' % mode)


def write_numpy_file(file_name, file_type, output_directory, scmap):
    file = os.path.join(output_directory, file_name + file_type)
    np.save(file, scmap)


def write_channel_pose_maps(file_name, file_type, original_image_shape, output_directory, scmap):
    for i in range(scmap.shape[-1]):
        file = os.path.join(output_directory, file_name + str(i) + file_type)
        resized = cv2.resize(heat_map_to_image(scmap[:, :, i]), (original_image_shape[1], original_image_shape[0]))
        cv2.imwrite(file, resized)


def write_combined_pose_map(file_name, file_type, original_image_shape, output_directory, scmap):
    file = os.path.join(output_directory, file_name + file_type)
    combined_map = pose_map_to_image(scmap)
    resized = cv2.resize(combined_map, (original_image_shape[1], original_image_shape[0]))
    cv2.imwrite(file, resized)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-pattern', help='Pattern to find the input images')
    parser.add_argument('--output-dir', help='Directory to write the poses to')
    parser.add_argument('--mode', help='output png images instead of numpy files', choices=['np', 'png', 'channel-pngs'], default='np')
    args = parser.parse_args()
    print(args)

    print('Searching files to be processed...')
    all_image_paths = glob.glob(args.input_pattern)
    print('Found %d files to be processed.' % len(all_image_paths))

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    calculate_and_write_pose_maps(all_image_paths, args.output_dir, args.mode)


if __name__ == '__main__':
    main()
