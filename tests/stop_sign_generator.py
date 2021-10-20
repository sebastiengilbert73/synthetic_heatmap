import logging
from synthetic_heatmap.generators.stop_sign import StopSign
import os
import cv2
import argparse
import ast

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main(
    numberOfImages,
    image_sizeHW,
    output_directory,
    displayResultMsg
    ):
    logging.info("stop_sign_generator.py main()")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    stop_sign_generator = StopSign()
    for imgNdx in range(numberOfImages):
        input_img, heatmap_img, result_msg = stop_sign_generator.Generate(image_sizeHW,
                                                                          maximum_number_of_trials=10,
                                                                          debug_directory=None)
        if displayResultMsg:
            logging.info(result_msg)
        input_img_filepath = os.path.join(output_directory, "stopSignGenerator_main_input{}.png".format(imgNdx))
        cv2.imwrite(input_img_filepath, input_img)
        heatmap_img_filepath = os.path.join(output_directory, "stopSignGenerator_main_heatmap{}.png".format(imgNdx))
        cv2.imwrite(heatmap_img_filepath, heatmap_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numberOfImages', help="The number of generated images. Default: 160", type=int, default=160)
    parser.add_argument('--imageSizeHW', help="The generated image size (height, width). Default: '(256, 256)'", default='(256, 256)')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs'", default='./outputs')
    parser.add_argument('--displayResultMsg', help="Display the result message, from the image download", action='store_true')
    args = parser.parse_args()

    image_sizeHW = ast.literal_eval(args.imageSizeHW)
    main(args.numberOfImages,
         image_sizeHW,
         args.outputDirectory,
         args.displayResultMsg)