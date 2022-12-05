# Formula Recognition

_The repo serves as a server node to recognize a math expression, transfer it into an expression tree,
evaluate the result and publish it to ROS topic._

The repository contains a ROS package, relying on 
`cv_bridge`, 
`geometry_msgs`, 
`intera_interface`, 
`rospy` and
`std_msgs`. 

There are two nodes here: 

### `src/intera_cam.py`

This node serve as an test node, with no topic or service interaction with the controller or
path planning part. It: 
* retrieves images from one of the Sawyer's cameras (by default `head_camera`, but testing shows the `right_hand_camera` on the wrist is better),
* feeds them into a pretrained model (defined in `src/model.py` whose trained parameters are indexed by the `model.lst`) to interpret the Latex expression,
* parse the expressions,
* evaluate results of the parsed expression, and
* respond to a client's query with the results.


### `src/server.py`

This node relies on `src/intera_cam.py`, shouldering almost the same functionalities but: 
* will not start CV recognition until a service request from `/formula/get_solution_int` or `/formula/get_solution_str`
* will run the recognition model continuously after a request is received until a specified number of answers is generated
* will return the service client with the **most frequent** answer among all the answers as many as the specified number
* will turns itself off to the silent mode and waiting for a new request again

## ROS Node details

### A. Preparation

After cloning the repository, the first thing is download the pretrained models. The downloading relies on
[Intel's Open Model Zoo Downloader](https://docs.openvino.ai/latest/omz_tools_downloader.html#doxid-omz-tools-downloader).
After the downloaded is installed, you may run 

```sh
omz_downloader --list model.lst
```

which will download all the necessary models under the package's root directory. Four models will be downloaded: 
recognizing handwriten or printed characters, and each contains both the FP32 and FP16 virations. The directory
storing all the models are named `intel`. For each model, there will be two `.XML` files, one for the encoder and
the other for the decoder, whose paths will be used to launch the node. 

Of course, never forget to `catkin_make` and `source devel/setup.bash`. 

Moreover, the Sawyer robotic arm should be started. Occasionally, the camera may fail to response. You may need to
first SSH into the Sawyer and re-enable the robot, and then **exit the SSH session** to launch the node again. 

### B. Launch file and parameters

#### `launch/start.launch`

This starts the `intera_cam.py` node, with the following customizable parameters: 
* **`encoder`**: path to the encoder model's XML file.
* **`decoder`**: path to the decoder model's XML file.
* **`vocab`**: path the decoder model's `vocab.json`.
* **`camera`**: specify which camera to use, either `head_camera` (default) or `right_hand_camera` (recommended).
* **`confidence`**: confidence level, below which the parsed expression will be discarded.
* **`preprocessing`**: the image preprocessing method, either `crop` or `resize` (default).

#### `launch/serve.launch`

This starts the `server.py`

* **`encoder`**: path to the encoder model's XML file.
* **`decoder`**: path to the decoder model's XML file.
* **`vocab`**: path the decoder model's `vocab.json`.
* **`camera`**: specify which camera to use, either `head_camera` or `right_hand_camera` (default and recommended).
* **`confidence`**: confidence level, below which the parsed expression will be discarded.
* **`preprocessing`**: the image preprocessing method, either `crop` (default) or `resize`.



### C. ROS Service

The ROS service requires a number as the request, which specifies how many confident answers the CV model should
produced until it returns a result. The result will be the most frequent one among the answers. There are two
service channels, differing only by the returned types: 

#### `/formula_rec/get_solution_int`

This returns an integral result. Should the answer is a floating point number, it will round the number
to the nearest integer. 

#### `/formula_rec/get_solution_str`

This returns a floating point number as a string. Two digits after decimal point will be preserved. 

## Logistics

Depending on Intel Open Model Zoo's models and sample implementation. 

Open-sourced under Apache licence. 
