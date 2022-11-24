#!/usr/bin/env python3
"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import sys
from argparse import SUPPRESS, ArgumentParser

import cv2 as cv
from utils import (PREPROCESSING, Model, calculate_probability, preprocess_image)

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.WARN, stream=sys.stdout)

def create_capture(input_source, demo_resolution):
    try:
        input_source = int(input_source)
    except ValueError:
        pass
    capture = cv.VideoCapture(input_source)
    capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, demo_resolution[0])
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, demo_resolution[1])
    return capture


def non_interactive_demo(model, args):
    _, _, height, width = model.encoder.input("imgs").shape
    target_shape = (height, width)
    vCap = create_capture(args.input, [width, height])
    while True:
        ret, frame = vCap.read()
        if not ret:
            break

        blur = cv.GaussianBlur(
            cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
            (5, 5), 0
        )
        th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY, 15, 6)
        th = cv.cvtColor(th, cv.COLOR_GRAY2BGR)
        cv.imshow("Display", th)
        # wait for 1ms or key press
        k = cv.waitKey(1) #k is the key pressed
        image = preprocess_image(PREPROCESSING[args.preprocessing_type],
            th, target_shape)

        distribution, targets = model.infer_sync(image)
        prob = calculate_probability(distribution)
        log.info("Confidence score is {}".format(prob))
        if prob >= args.conf_thresh ** len(distribution):
            phrase = model.vocab.construct_phrase(targets)
            print(f"Formula: {phrase}\n")
        if k == 27 or k == 113:  #27, 113 are ascii for escape and q respectively
            break
    vCap.release()
    cv.destroyAllWindows()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help',
                      default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m_encoder", help="Required. Path to an .xml file with a trained encoder part of the model",
                      required=True, type=str)
    args.add_argument("-m_decoder", help="Required. Path to an .xml file with a trained decoder part of the model",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Integer identifier of the camera or path to the video",
                      required=True, type=str)
    args.add_argument("-v", "--vocab_path", help="Required. Path to vocab file to construct meaningful phrase",
                      type=str, required=True)
    args.add_argument("--max_formula_len",
                      help="Optional. Defines maximum length of the formula (number of tokens to decode)",
                      default="128", type=int)
    args.add_argument("-t", "--conf_thresh",
                      help="Optional. Probability threshold to treat model prediction as meaningful",
                      default=0.95, type=float)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. Default value "
                           "is CPU",
                      default="CPU", type=str)
    args.add_argument("--resolution", default=(1280, 720), type=int, nargs=2,
                      help='Optional. Resolution of the demo application window. Default: 1280 720')
    args.add_argument('--preprocessing_type', choices=PREPROCESSING.keys(),
                      help="Optional. Type of the preprocessing", default='crop')
    args.add_argument('--imgs_layer', help='Optional. Encoder input name for images. See README for details.',
                      default='imgs')
    args.add_argument('--row_enc_out_layer',
                      help='Optional. Encoder output key for row_enc_out. See README for details.',
                      default='row_enc_out')
    args.add_argument('--hidden_layer', help='Optional. Encoder output key for hidden. See README for details.',
                      default='hidden')
    args.add_argument('--context_layer', help='Optional. Encoder output key for context. See README for details.',
                      default='context')
    args.add_argument('--init_0_layer', help='Optional. Encoder output key for init_0. See README for details.',
                      default='init_0')
    args.add_argument('--dec_st_c_layer', help='Optional. Decoder input key for dec_st_c. See README for details.',
                      default='dec_st_c')
    args.add_argument('--dec_st_h_layer', help='Optional. Decoder input key for dec_st_h. See README for details.',
                      default='dec_st_h')
    args.add_argument('--dec_st_c_t_layer', help='Optional. Decoder output key for dec_st_c_t. See README for details.',
                      default='dec_st_c_t')
    args.add_argument('--dec_st_h_t_layer', help='Optional. Decoder output key for dec_st_h_t. See README for details.',
                      default='dec_st_h_t')
    args.add_argument('--output_layer', help='Optional. Decoder output key for output. See README for details.',
                      default='output')
    args.add_argument('--output_prev_layer',
                      help='Optional. Decoder input key for output_prev. See README for details.',
                      default='output_prev')
    args.add_argument('--logit_layer', help='Optional. Decoder output key for logit. See README for details.',
                      default='logit')
    args.add_argument('--tgt_layer', help='Optional. Decoder input key for tgt. See README for details.',
                      default='tgt')
    return parser


def main():
    args = build_argparser().parse_args()
    model = Model(args, False)
    non_interactive_demo(model, args)


if __name__ == '__main__':
    sys.exit(main() or 0)
