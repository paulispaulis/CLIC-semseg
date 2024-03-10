# Documentation and ROS Integration 

This forlder includes:

- The paper describing CLIC approach to the semantic segmentation
- Documented code version in the .zip file (incomplete)
- Drawing summarizing the inference process
- Robot Operating System (ROS) integration Python script sample.

## Dockerfile Recommendations:

- Base image: ROS official Docker image.
- Install dependencies: Python, PyTorch, CUDA (if using GPU), and the segmentation framework.
- Set up a ROS workspace, clone your ROS package, and build it.
- Set the entry point to initialize the ROS environment.

## Handling Framerate and Avoiding Buildup of Unprocessed Frames:

Queue Size: The queue_size=1 parameter in both the subscriber and publisher ensures that only the most recent messages are kept. If the processing node can't keep up, older frames are dropped.

Buffer Size: The buff_size=2**24 parameter helps to handle the buffering of large messages, like high-resolution images.


