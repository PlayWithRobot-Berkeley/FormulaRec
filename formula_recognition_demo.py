#!/usr/bin/env python3
import sys
from typing import Callable, Tuple, Any, List, Dict

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import intera_interface
from utils import (PREPROCESSING, Model, calculate_probability, preprocess_image)
import rospy

from evaluate import evaluate_exp

def image_callback(img_data, handlers: List[Callable[[cv.Mat], None]]):
    """The callback function to retrieve image by using CvBridge

    Params
    ------
    img_data: the image topic data
    handlers: a list (or tuple) of Callable taking in the image from camera
    """
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")
    except CvBridgeError as err:
        rospy.logerr(err)
        return
    if handlers: 
        for handler in handlers:
            handler(cv_image)
    cv.waitKey(1)


def formula_recognizer(model: Model, target_shape: Tuple[int, int], args: Dict[str, Any], frame: cv.Mat): 
    """ Recognize image using the non-interactive mode of the model

    Params
    ------
    model: the Model to be used
    target_shape: a tuple of height, width required by the model as inputs
    args: key-value arguments
    frame: the image from camera
    """
    blur = cv.GaussianBlur(
        cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
        (5, 5), 0
    )
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 15, 6)
    th = cv.cvtColor(th, cv.COLOR_GRAY2BGR)
    cv.imshow("Display", th)
    image = preprocess_image(PREPROCESSING[args['preprocessing_type']],
        th, target_shape)

    distribution, targets = model.infer_sync(image)
    prob = calculate_probability(distribution)
    rospy.loginfo("Confidence score is {}".format(prob))
    if prob >= args['conf_thresh'] ** len(distribution):
        phrase = model.vocab.construct_phrase(targets)
        try:
            print(f"Formula: {phrase} = {evaluate_exp(phrase)}\n")
        except ValueError:
            print(f"Formula: {phrase} -> incorrect, skipped")
        except Exception as e:
            print(f"Formula: {phrase} -> {type(e)}: e")


def load_parameters() -> Dict[str, Any]:
    """ Read arguments from ROS parameter server

    Returns
    -------
    A dictionary, whose keys are the parameter names, and the 
    values are the arguments
    """
    args = {}
    # REQUIRED param
    if not rospy.has_param('~m_encoder'):
        raise ValueError('m_encoder parameter must be specified')
    args['m_encoder'] = rospy.get_param('~m_encoder')
    if not rospy.has_param('~m_decoder'):
        raise ValueError('m_decoder parameter must be specified')
    args['m_encoder'] = rospy.get_param('~m_decoder')
    if not rospy.has_param('~vocab_path'):
        raise ValueError('vocab_path parameter must be specified')
    args['vocab_path'] = rospy.get_param('~vocab_path')

    # Optional param
    def optional_param_getter(name: str, default_val): 
        args[name] = default_val if not rospy.has_param(f'~{name}') \
            else rospy.get_param(f'~{name}')
    optional_param_getter('camera', 'head_camera')
    optional_param_getter('conf_thresh', 0.95)
    optional_param_getter('device', 'CPU')
    optional_param_getter('max_formula_len', 128)
    optional_param_getter('resolution', (1280, 720))
    optional_param_getter('imgs_layer', 'imgs')
    optional_param_getter('row_enc_out_layer', 'row_enc_out')
    optional_param_getter('hidden_layer', 'hidden')
    optional_param_getter('context_layer', 'context')
    optional_param_getter('init_0_layer', 'init_0')
    optional_param_getter('dec_st_c_layer', 'dec_st_c')
    optional_param_getter('dec_st_h_layer', 'dec_st_h')
    optional_param_getter('dec_st_c_t_layer', 'dec_st_c_t')
    optional_param_getter('dec_st_h_t_layer', 'dec_st_h_t')
    optional_param_getter('output_layer', 'output')
    optional_param_getter('output_prev_layer', 'output_prev')
    optional_param_getter('logit_layer', 'logit')
    optional_param_getter('tgt_layer', 'tgt')
    optional_param_getter('preprocessing', 'crop')

    if args['preprocessing'] not in PREPROCESSING: 
        raise ValueError(f'unrecognized preprocessing argument: {args["preprocessing"]}')
    
    return args
    
        

def main():
    # STEP ONE: init
    rp = intera_interface.RobotParams()
    valid_cameras = rp.get_camera_names()
    if not valid_cameras:
        rp.log_message(("Cannot detect any camera_config"
            " parameters on this robot. Exiting."), "ERROR")
        return 1
    
    rospy.init_node('recognize_formula_from_camera', anonymous=True)
    args = load_parameters()

    # STEP TWO: establish the model
    model = Model(args, False)
    _, _, height, width = model.encoder.input("imgs").shape
    input_shape = (height, width)

    # STEP THREE: set up the camera
    camera = intera_interface.Cameras()
    if not camera.verify_camera_exists(args['camera']):
        rospy.logerr("Invalid camera name. Exiting")
        return 2
    recognizer_closure = lambda frame: formula_recognizer(model, input_shape, args, frame)
    camera.set_callback(args['camera'], image_callback, 
        rectify_image = True, callback_args=(recognizer_closure))

    # STEP FOUR: start
    rospy.on_shutdown(cv.destroyAllWindows)
    rospy.loginfo("Camera_display node running. Ctrl-c to quit")
    rospy.spin()


if __name__ == '__main__':
    sys.exit(main() or 0)
