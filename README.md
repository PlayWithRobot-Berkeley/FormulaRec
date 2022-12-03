# Formula Recognition

_The repo serves as a server node to recognize a math expression, transfer it into an expression tree,
evaluate the result and publish it to ROS topic._

The repository contains a ROS package, relying on 
`cv_bridge`, 
`geometry_msgs`, 
`intera_interface`, 
`rospy` and
`std_msgs`. 
The node to be used is the `src/intera_cam.py`. It
* retrieves images from one of the Sawyer's cameras (by default `head_camera`, but testing shows the `right_hand_camera` on the wrist is better),
* feeds them into a pretrained model (defined in `src/model.py` whose trained parameters are indexed by the `model.lst`) to interpret the Latex expression,
* parse the expressions,
* evaluate results of the parsed expression, and
* respond to a client's query with the results.

## ROS Node details

### A. Preparation

After cloning the repository, the first thing is download the pretrained models. The downloading relies on
[Intel's Open Model Zoo Downloader](https://docs.openvino.ai/latest/omz_tools_downloader.html#doxid-omz-tools-downloader).
After the downloaded is installed, you may run 

```sh
omz_downloader model.lst
```

which will download all the necessary models under the package's root directory. Four models will be downloaded: 
recognizing handwriten or printed characters, and each contains both the FP32 and FP16 virations. The directory
storing all the models are named `intel`. For each model, there will be two `.XML` files, one for the encoder and
the other for the decoder, whose paths will be used to launch the node. 

Of course, never forget to `catkin_make` and `source devel/setup.bash`. 

Moreover, the Sawyer robotic arm should be started. Occasionally, the camera may fail to response. You may need to
first SSH into the Sawyer and re-enable the robot, and then **exit the SSH session** to launch the node again. 

### B. Launch file and parameters

The launch file locates in `launch/start.launch`, with the following customizable parameters: 
* **`encoder`**: path to the encoder model's XML file.
* **`decoder`**: path to the decoder model's XML file.
* **`vocab`**: path the decoder model's `vocab.json`.
* **`camera`**: specify which camera to use, either `head_camera` (default) or `right_hand_camera` (recommended).
* **`confidence`**: confidence level, below which the parsed expression will be discarded.
* **`preprocessing`**: the image preprocessing method, either `crop` or `resize`.


### C. ROS Service

__*TODO*__

The node will not start recognition immediately it is started. It will be silent until a request is sent
from the client. It will regard the request as a signal indicating "the question board is set up already". 
Thus, it is desirable that the client will not sent the request until it is ready. 

The node will then capture 100 frames from the camera, parsing each and finding the result appears the most
among the 100 frames. A responds carrying the result will thereafter be dispatched. 

## Logistics

Depending on Intel Open Model Zoo's models and sample implementation. 

Open-sourced under Apache licence. 
