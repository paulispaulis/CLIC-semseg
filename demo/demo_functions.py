"""This file contains functions for image and video processing using semantic segmentation models (demo)."""

import sys
sys.path.append('..')
from src import imageops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
import streamlit as st
import cv2


colorsl = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0], [255, 255, 255], [128, 128, 128], [128, 0, 0], [0, 128, 0], [0, 0, 128]]
colors = np.array(colorsl)
colorsl = [[c / 255 for c in color] for color in colorsl]

def array_to_rgb(array):
    """
    Converts a multi-channel array to an RGB image.

    Args:
        array (numpy.ndarray): Input multi-channel array.

    Returns:
        numpy.ndarray: RGB image array.
    """

    n, y, x = array.shape
    max_indices = np.argmax(array, axis=0)
    rgb_image = np.zeros((y, x, 3), dtype=np.uint8)
    for i in range(n):
        mask = max_indices == i
        rgb_image[mask] = colors[i]

    return rgb_image


def run_image(smodel, path, labels, multilabel = False, aggregation = 'max', rough_labels = None, web = False):
    """
    Runs semantic segmentation on a images and displays the result.

    Args:
        smodel (object): Semantic segmentation model.
        path (str): Path to the input image.
        labels (list): List of labels to be segmented.
        multilabel (bool, optional): Whether multilabel segmentation is used. Defaults to False.
        aggregation (str, optional): Aggregation method for multilabel segmentation. Defaults to 'max'.
        rough_labels (list, optional): Rough labels for multilabel segmentation. Defaults to None.
        web (bool, optional): Whether to display the result in a web interface. Defaults to False.
    """

    img = imageops.open_image(path)
    if multilabel == False:
        hmaps = smodel.forward(img, labels)
    else:
        hmaps = smodel.forward_multilabel(img, labels, aggregation = aggregation)

    fig = plt.figure(figsize=(28, 8))

    width = ((len(labels) + 2 + 1) // 2) * 2
    gs = gridspec.GridSpec(2, width, width_ratios=[1, 0.05]*(width // 2))

    # Original image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, aspect = 'auto')
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.imshow(array_to_rgb(hmaps), aspect = 'auto')
    ax1.set_xticks([])
    ax1.set_yticks([])

    if multilabel:
        if rough_labels is None:
            labels_s = [l[0] for l in labels]
        else:
            labels_s = rough_labels
    else:
        labels_s = labels

    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colorsl[:len(labels_s)], labels_s)]
    ax1.legend(handles=legend_patches)

    # Individual heatmaps and their colorbars
    for idx, l in enumerate(labels):
        ax = fig.add_subplot(gs[(idx * 2 + 4) // width, (idx*2 + 4) % width])
        cax = fig.add_subplot(gs[(idx * 2 + 5) // width, (idx*2 + 5) % width])
        pos = cax.get_position()
        new_pos = [pos.x0 - 0.01, pos.y0, pos.width, pos.height]
        cax.set_position(new_pos)
        ax.set_xticks([])
        ax.set_yticks([])

        im = ax.imshow(hmaps[idx], aspect = 'auto', vmin = 0, vmax = 1)
        plt.colorbar(im, cax=cax, pad = 0)
        ax.set_title(labels_s[idx])

    if web:
        st.pyplot(fig)
    else:
        plt.savefig(path[:-4] + 'output' + '.png', bbox_inches = 'tight')


def run_video(path, smodel, labels, width, height, output_path, fps = 30, frame_count = 1000, frame_average = 5, draw_period = 10, web = False):
    """
    Runs semantic segmentation on a video and outputs the processed video.

    Args:
        path (str): Path to the input video.
        smodel (object): Semantic segmentation model.
        labels (list): List of labels to be segmented.
        width (int): Width of the output video.
        height (int): Height of the output video.
        output_path (str): Path to save the processed video.
        fps (int, optional): Frames per second of the output video. Defaults to 30.
        frame_count (int, optional): Maximum number of frames to process. Defaults to 1000.
        frame_average (int, optional): Number of frames to average over for smoother segmentation. Defaults to 5.
        draw_period (int, optional): Interval for displaying intermediate results. Defaults to 10.
        web (bool, optional): Whether to display intermediate results in a web interface. Defaults to False.
    """

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    video_reader = VideoReader(path)

    hmaps = []
    for i in range(frame_count):
        frame = video_reader.get_frame() # Get next video frame
        if frame is -1:
            break
        frame = frame[:height, :width]

        #There's prolly a problem with RGB BGR.
        hmaps.append(run_image_vid(smodel, frame, labels))

        #Averaging over previous heatmaps
        if len(hmaps) > frame_average:
            hmaps = hmaps[1:]
        avg = np.array(hmaps).mean(axis = 0)
        res_frame = array_to_rgb(avg)

        out.write(res_frame // 2 + frame // 2) #Writing to video file

        #spam
        if i % draw_period is 0 and not web:
            legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colorsl[:len(labels)], labels)]
            print('On frame ', i, '/', frame_count)
            plt.figure()
            plt.imshow(res_frame)
            plt.legend(handles = legend_patches)
            plt.title('frame ' + str(i))

    video_reader.release()

    out.release()


def run_image_vid(smodel, img, labels):
    """
    Inference on a single image.

    Args:
        smodel (object): Semantic segmentation model.
        img (numpy.ndarray): Input image.
        labels (list): List of labels to be segmented.

    Returns:
        numpy.ndarray: Segmented image.
    """

    hmaps = smodel.forward(img, labels)
    return hmaps


class VideoReader:
    """Video reader class for reading frames from a video file."""

    def __init__(self, video_path):
        """
        Initializes the VideoReader.

        Args:
            video_path (str): Path to the video file.
        """

        self.cap = cv2.VideoCapture(video_path)

    def get_frame(self):
        """
        Reads the next frame from the video.

        Returns:
            numpy.ndarray: The next frame, or -1 if end of video is reached.
        """

        ret, frame = self.cap.read()
        if not ret:
            return -1
        return frame

    def release(self):
        """Releases the video capture."""

        self.cap.release()