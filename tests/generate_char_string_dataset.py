import logging
from synthetic_heatmap.generators.char_string import CharString
import os
import cv2
import argparse
import ast

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main(
        imageSizeHW,
        outputDirectory,
        numberOfImages
):
    logging.info("generate_char_string_dataset.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Create a generator
    generator = CharString()

    for imageNdx in range(numberOfImages):
        image, heatmap = generator.Generate(
            image_sizeHW=imageSizeHW,
            debug_directory=outputDirectory
        )
        cv2.imwrite(os.path.join(outputDirectory, "image_{}.png".format(imageNdx)), image)
        cv2.imwrite(os.path.join(outputDirectory, "heatmap_{}.png".format(imageNdx)), heatmap)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageSizeHW', help="The generated image size (height, width). Default: '(256, 256)'",
                        default='(256, 256)')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs_generate_char_string_dataset'", default='./outputs_generate_char_string_dataset')
    parser.add_argument('--numberOfImages', help="The number of generated images. Default: 256", type=int, default=256)
    args = parser.parse_args()

    imageSizeHW = ast.literal_eval(args.imageSizeHW)
    main(
        imageSizeHW,
        args.outputDirectory,
        args.numberOfImages
    )