# Documentation for ROS Integration 

Above root folder contains the actual code used for training the CLIC semantic segmentation.
This forlder instead focuses on using the trained CLIC for inference and includes:

- The paper describing CLIC approach to the semantic segmentation
- Self-contained notebook illustrating image segmentation (works in Google Colab)
- Documented code version in the .zip file (incomplete)
- Drawing summarizing the inference process
- Robot Operating System (ROS) integration Python script sample.

## Dockerfile Recommendations:

- Base image: ROS official Docker image.
- Install dependencies: Python, PyTorch, CUDA (if using GPU), and the segmentation framework.
- Set up a ROS workspace, clone your ROS package, and build it.
- Set the entry point to initialize the ROS environment.
- Run the container, ensuring it's connected to the ROS network (using network settings or ROS_MASTER_URI).
- Root folder contains an example Dockerfile.txt for Streamlit demo. A self-contained pre-build docker image can be run with command "docker run -it -p 8503:8501 --gpus all guntisb/clic:commit2". It can serve as a template for the ROS docker integration.

## Handling Framerate and Avoiding Buildup of Unprocessed Frames:

Queue Size: The queue_size=1 parameter in both the subscriber and publisher ensures that only the most recent messages are kept. If the processing node can't keep up, older frames are dropped.

Buffer Size: The buff_size=2**24 parameter helps to handle the buffering of large messages, like high-resolution images.


