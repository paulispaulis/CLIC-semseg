# From Captions to Pixels: Open-Set Semantic Segmentation without Masks

This repository contains the code implementation of the findings presented in the paper [From Captions to Pixels: Open-Set Semantic Segmentation without Masks](https://www.bjmc.lu.lv/contents/papers-in-production/).

## Authors:

* Paulis Barzdins
* Ingus Pretkalnins
* Guntis Barzdins

This reserach received 2024 Latvia Science Award as part of the integration into the real robotic systems "Robota kognitīva uztvere un augsta līmeņa instrukciju interpretācija ar dabiskās valodas jēdzieniem" ([announcementEN] (https://www.lza.lv/en/activities/news/2120-the-latvian-academy-of-sciences-announces-winners-of-the-annual-science-achievements-competition-2024) | [announcementLV](https://www.lza.lv/aktualitates/lza-balva/2118-2024-gada-sasniegumi) | [description](https://github.com/paulispaulis/CLIC-semseg/blob/main/Documentation/sasniegumi_2024_final.pdf) | [video](https://replay.lsm.lv/lv/skaties/ieraksts/ltv/348077/latvija-radits-robots-kurs-saprot-cilveka-izteiktas-komandas)).

## Overview

This paper presents a novel approach to open-set semantic segmentation in unstructured environments where there are no meaningful prior mask proposals. Our method leverages pre-trained encoders from foundation models and uses image-caption datasets for training, reducing the need for annotated masks and extensive computational resources. We introduce a novel contrastive loss function, named CLIC (Contrastive Loss function on Image-Caption data), which enables training a semantic segmentation model directly on an image-caption dataset. By utilising image-caption datasets, our method provides a practical solution for semantic segmentation in scenarios where large-scale segmented mask datasets are not readily available, as is the case for unstructured environments where full segmentation is unfeasible. Our approach is adaptable to evolving foundation models, as the encoders are used as black-boxes. The proposed method has been designed with robotics applications in mind to enhance their autonomy and decision-making capabilities in real-world scenarios.

## 🧪 Explore the Results Yourself!

### There are 2 ways to test the models both locally and on Google Colab:

#### Locally

- #### Streamlit demo locally

    Step 1: Clone the repository

    ``` !git clone https://github.com/paulispaulis/clip-schizo.git ```

    Step 2: Install FFmpeg

    Step 3: Install dependencies

    ``` pip install -r requirements.txt ```

    Step 4: Navigate to the demo folder

    Step 5: Run the command in the terminal

    ``` python -m streamlit run streamlit_demo.py ```

- #### Follow the notebook on your local machine

    Step 1: Navigate to [demo_notebook.ipynb](https://github.com/paulispaulis/CLIC-semseg/blob/main/demo/demo_notebook.ipynb).

    Step 2: Follow the notebook and see awesome segmentation results! 🔥

#### Google Colab

- #### Streamlit demo on Google Colab

    Step 1: [Head to Google Colab to test out the Streamlit demo](https://colab.research.google.com/drive/1IItKT7UV0fU_rYPxNKs05br9y-oT1ltF?usp=sharing)

    Step 2: Follow the notebook and launch the Streamlit demo! 🔥

- #### Follow the notebook on Google Colab

    Step 1: [Head to Google Colab to test the segmentation models](https://colab.research.google.com/drive/1720bBDth233E8L_AHoPB9n0JyRWUtXFd?usp=sharing)

    Step 2: Follow the notebook and see awesome segmentation results! 🔥

## Acknowledgments

This research is funded by the Latvian Council of Science project “Smart Materials, Photonics, Technologies and Engineering Ecosystem” project No VPP-EM-FOTONIKA-2022/1-0001 and by the Latvian Council of Science project lzp-2021/1-0479.
