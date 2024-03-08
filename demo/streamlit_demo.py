import os
import sys
sys.path.append('..')
from src import nns
from src import embedders
from src import segmodel
import torch
from demo_functions import run_image, run_video
import streamlit as st
from src import defdevice
import os
import tempfile
st.set_page_config(layout="wide")

if torch.cuda.is_available():
    defdevice.force_device('cuda:0')

image_embr = embedders.M2FImageEmbedder()
text_embr = embedders.CLIPTextEmbedder()
model = nns.Linear(1, 1)

st.header('From Captions to Pixels: Open-Set Semantic Segmentation without Masks')
choice = st.radio("Choose processing type", ('Image', 'Video'))
if choice == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type = ["jpg", "jpeg", "png"])
    input_labels = st.text_input("Input labels", value = 'tree, dirt road')
    st.write("The chosen labels are", input_labels)
    option = st.selectbox("Choose model", ('CSTableModel', 'CSRUGDFineTuned'))

    if uploaded_file is not None and input_labels is not None:
        model.load(f"../weights/{option}")
        smodel = segmodel.CSModel(image_embr, text_embr, model)

        labels = []
        for label in input_labels.split(","):
            labels.append(label.strip())

        with st.spinner('Generating...'):
            run_image(smodel, uploaded_file, labels, web = True)

elif choice == "Video":
    uploaded_file = st.file_uploader("Choose a video...", type="mp4")
    input_labels = st.text_input("Input labels", value = 'mountain, moss, grass, rock, other')
    st.write("The chosen labels are", input_labels)

    if uploaded_file is not None and input_labels is not None:
        model.load("../weights/CSTableModel")
        smodel = segmodel.CSModel(image_embr, text_embr, model)

        labels = []
        for label in input_labels.split(","):
            labels.append(label.strip())

        # This is to go around the UploadedFile format
        if uploaded_file:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())

        # Define run_video functio parameters
        width, height = (468, 256)
        fps = 30
        frame_count = 50000 # Maximum amount of frames to segment.
        frame_average = 5 # How many frames to average segmentation over.
        draw_period = 300 # How frequently to output example segmentations.
        output_path = '../FP/generated_video.mp4'

        with st.spinner('Generating...'):
            run_video(
                path, smodel, labels, width, height, output_path, fps, frame_count, frame_average, draw_period, web = True)

            video_file = open(output_path, 'rb')
            video_bytes = video_file.read()

            st.video(video_bytes)
