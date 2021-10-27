import logging
import argparse
import torch
import os
import cv2
import ast
import image_to_heatmap_blurs

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

image_extensions = ['.JPG', '.PNG', '.TIF']

def main(
        imagesDirectory,
        neuralNetworkFilepath,
        device,
        architecture,
        outputDirectory,
        imageSizeHW,
        blurringSizes
):
    logging.info("batch_image_to_heatmap_blurs.py main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # List the images in the directory
    original_image_filepaths = [os.path.join(imagesDirectory, f) for f in os.listdir(imagesDirectory) if
                 os.path.isfile(os.path.join(imagesDirectory, f))
                 and f.upper()[-4:] in image_extensions]
    #logging.debug("original_image_filepaths = {}".format(original_image_filepaths))

    for filepath in original_image_filepaths:
        color_coded_img = image_to_heatmap_blurs.main(
            imageFilepath=filepath,
            neuralNetworkFilepath=neuralNetworkFilepath,
            device=device,
            architecture=architecture,
            outputDirectory=outputDirectory,
            image_sizeHW=imageSizeHW,
            blurring_sizes=blurringSizes,
            targetHeatmapFilepath=None
        )
        color_coded_img_filepath = os.path.join(outputDirectory, os.path.basename(filepath))
        cv2.imwrite(color_coded_img_filepath, color_coded_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imagesDirectory', help="The directory containing the images")
    parser.add_argument('neuralNetworkFilepath', help="The filepath to the neural network")
    parser.add_argument('--useCpu', help='Use cpu, even if cuda is available', action='store_true')
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'FConv_3k5_max_32_64_128'",
                        default='FConv_3k5_max_32_64_128')
    parser.add_argument('--outputDirectory',
                        help="The directory where the output data will be written. Default: './outputs_batchImageToHeatmapBlurs'",
                        default='./outputs_batchImageToHeatmapBlurs')
    parser.add_argument('--imageSizeHW', help="The image size (height, width). Default: '(512, 512)'",
                        default='(512, 512)')
    parser.add_argument('--blurringSizes', help="The blurring sizes. Default: '[1, 3]'", default='[1, 3]')
    args = parser.parse_args()

    device = 'cpu'
    useCuda = not args.useCpu and torch.cuda.is_available()
    if useCuda:
        device = 'cuda'
    imageSizeHW = ast.literal_eval(args.imageSizeHW)
    blurringSizes = ast.literal_eval(args.blurringSizes)

    main(
        args.imagesDirectory,
        args.neuralNetworkFilepath,
        device,
        args.architecture,
        args.outputDirectory,
        imageSizeHW,
        blurringSizes
    )