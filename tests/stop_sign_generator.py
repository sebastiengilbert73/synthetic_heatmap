import logging
from synthetic_heatmap.generators.stop_sign import StopSign
import os
import cv2

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

output_directory = './outputs'

def main():
    logging.info("stop_sign_generator.py main()")
    image_sizeHW = (256, 256)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    stop_sign_generator = StopSign()
    for imgNdx in range(20):
        input_img, heatmap_img = stop_sign_generator.Generate(image_sizeHW)
        input_img_filepath = os.path.join(output_directory, "stopSignGenerator_main_input{}.png".format(imgNdx))
        cv2.imwrite(input_img_filepath, input_img)
        heatmap_img_filepath = os.path.join(output_directory, "stopSignGenerator_main_heatmap{}.png".format(imgNdx))
        cv2.imwrite(heatmap_img_filepath, heatmap_img)

if __name__ == '__main__':
    main()