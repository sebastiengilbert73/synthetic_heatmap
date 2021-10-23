import logging
from synthetic_heatmap.generators.stop_sign import StopSign
import os
import cv2
import argparse
import ast

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main(
        imageSizeHW,
        outputDirectory,
        backgroundImageFilepath,
        word
):
    logging.info("generate_single_stop_sign.py main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
    background_img = None
    if backgroundImageFilepath is not None:
        background_img = cv2.imread(backgroundImageFilepath, cv2.IMREAD_GRAYSCALE)
    print ("background_img = {}".format(background_img))
    stop_sign_generator = StopSign()
    input_img, heatmap_img, result_msg = stop_sign_generator.Generate(imageSizeHW,
                                                                      maximum_number_of_trials=10,
                                                                      debug_directory=outputDirectory,
                                                                      background_image=background_img,
                                                                      word=word
                                                                      )
    input_img_filepath = os.path.join(outputDirectory, "generateSingleStopSign_main_input.png")
    cv2.imwrite(input_img_filepath, input_img)
    heatmap_img_filepath = os.path.join(outputDirectory, "generateSingleStopSign_main_heatmap.png")
    cv2.imwrite(heatmap_img_filepath, heatmap_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageSizeHW', help="The generated image size (height, width). Default: '(256, 256)'",
                        default='(256, 256)')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs'", default='./outputs')
    parser.add_argument('--backgroundImageFilepath', help="The filepath to the background image. Default: 'None'", default='None')
    parser.add_argument('--word', help="The word to put in the middle of the octagon. Default: 'STOP'", default='STOP')
    args = parser.parse_args()

    imageSizeHW = ast.literal_eval(args.imageSizeHW)
    backgroundImageFilepath = args.backgroundImageFilepath
    if args.backgroundImageFilepath.upper() == 'NONE':
        backgroundImageFilepath = None

    main(
        imageSizeHW,
        args.outputDirectory,
        backgroundImageFilepath,
        args.word
    )