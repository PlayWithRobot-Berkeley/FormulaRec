from typing import Callable, Union, Any, Dict
from threading import Lock

import intera_interface
import rospy

from formula_rec.srv import GetSolutionInt, GetSolutionIntRequest, GetSolutionStr, GetSolutionStrRequest
from intera_cam import image_callback, formula_recognizer, show_image, load_parameters
from utils import Model

class Serverlet:
    """ The server-side codes for the service between the controller and the CV

    The server responds to the corresponding service requests, starts the CV codes
    and returns the most frequent results.

    Params
    ------
    args: the rospy parameter
    model: the CV model
    cameras: the intera_interface.Cameras
    """
    def __init__(self, args: Dict[str, Any], model: Model, cameras: intera_interface.Cameras):
        self._model = model
        self._args = args
        self._cameras = cameras

        self._answer_cnt = 0
        """How many answers the CV model has produced since it is reset"""
        self._answer_2_cnt: Dict[Union[float, int], int] = {}
        """Among the answers, each one's count"""
        self._answer_lock = Lock()
        """A lock to protect the answer counter and the dictionary bookkeeping"""
        self._camera_name = args["camera"]

        _, _, height, width = model.encoder.input("imgs").shape
        self._input_shape = (height, width)
        """The desirable image shape for the CV model"""

        if not self._cameras.verify_camera_exists(self._camera_name):
            rospy.logerr(f"Invalid camera name: {self._camera_name}. Exiting")
            rospy.signal_shutdown()
        self.reset()

    def reset(self):
        """Reset the answer counter and bookkeeping, and the camera callback
        
        After it is reset and before it starts to handling a request, the server
        only display the images.
        """
        self._cameras.set_callback(self._camera_name, image_callback, 
            rectify_image = True, callback_args=(show_image, ))
        # only one handler is provided as the image_callback's args
        # => The image will be displayed, but the model will not be 
        # called until a new request comes 
        try: 
            self._answer_lock.acquire()
            self._answer_2_cnt.clear()
            self._answer_cnt = 0
        finally:
            self._answer_lock.release()
    
    def _recognizer_closure(self, frame):
        """ Contradictory to the init state after the reseting, when a request
        is being handled, a frame from the camera will be recongnized whose answer
        will be stored in the bookkeeping dictionary. 

        Params
        ------
        frame: the frame to be recognized
        """
        result = formula_recognizer(self._model, self._input_shape, self._args, frame)
        if result is None:
            return
        try: 
            self._answer_lock.acquire()
            self._answer_cnt += 1
            if result not in self._answer_2_cnt:
                self._answer_2_cnt[result] = 0
            self._answer_2_cnt[result] += 1
        finally:
            self._answer_lock.release()

    
    def _wait_until_answer_queue_is_full(self, num: int) -> Union[float, int]:
        """ Block the current procedure until a required number of answers has
        been collected. Once the number is reached, the most frequent one will
        be returned after this Serverlet is reset.

        Params
        ------
        num: how many answer is required

        Returns
        -------
        The numerical result
        """
        rate = rospy.Rate(1)
        while True: 
            try:
                self._answer_lock.acquire()
                if self._answer_cnt >= num: 
                    max_cnt, max_cnt_val = 0, 0
                    for val, cnt in self._answer_2_cnt.items():
                        if cnt > max_cnt: max_cnt, max_cnt_val = cnt, val
                    self.reset()
                    return max_cnt_val
                else:
                    rate.sleep()
            finally:
                self._answer_lock.release()


    def _inner_callback(self, request: Union[GetSolutionIntRequest, GetSolutionStrRequest]) -> Union[float, int]: 
        """ Handling a request. 

        This is a general inner method, regardless what respond
        type is expected.
        
        NOTE
        ----
        this method will block the current procedure until a sufficient
        number of frames has been recognized and evaluated. 

        Params
        ------
        request: service request

        Returns
        -------
        the numerical result
        """
        num_of_tries = request.number_of_tries
        self._cameras.set_callback(self._camera_name, image_callback, 
            rectify_image = True, callback_args=(show_image, self._recognizer_closure))
        return self._wait_until_answer_queue_is_full(num_of_tries)


    def get_solution_int_callback(self, request: GetSolutionIntRequest) -> int: 
        """Service callback for `/formula_rec/get_solution_int` whose client
        expects an integral respond.
        """
        rospy.wait_for_service(f'/formula_rec/get_solution_int')
        return int(self._inner_callback(request))
    
    def get_solution_str_callback(self, request: GetSolutionStrRequest) -> str:
        """Service callback for `/formula_rec/get_solution_str` whose client
        expects a string as the respond.
        """
        rospy.wait_for_service(f'/formula_rec/get_solution_str')
        return f"{self._inner_callback(request)}:.2f"


    def register(self):
        """Register its two callbacks to ROS service"""
        rospy.Service(
            '/formula_rec/get_solution_int',
            GetSolutionInt,
            self.get_solution_int_callback
        )
        rospy.Service(
            '/formula_rec/get_solution_str',
            GetSolutionStr,
            self.get_solution_str_callback
        )
        rospy.loginfo('Services registered. The server is running...')
        
if __name__ == '__main__':
    rp = intera_interface.RobotParams()
    valid_cameras = rp.get_camera_names()
    if not valid_cameras:
        rp.log_message(("Cannot detect any camera_config"
            " parameters on this robot. Exiting."), "ERROR")
        exit(1)
    
    rospy.init_node('recognize_formula_per_request', anonymous=True)
    args = load_parameters()
    model = Model(args, False)
    cameras = intera_interface.Cameras()

    serverlet = Serverlet(args, model, cameras)
    serverlet.register()
    rospy.spin()
