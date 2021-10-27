import logging
import argparse
import torch
import os
import cv2
import ast
import image_to_heatmap_blurs
import imageio

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def main(
        neuralNetworksDirectory,
        imageFilepath,
        device,
        architecture,
        outputDirectory,
        imageSizeHW,
        blurringSizes
):
    logging.info("animation_multiNN.py main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # List the neural networks in the directory
    neural_network_filepaths = [os.path.join(neuralNetworksDirectory, f) for f in os.listdir(neuralNetworksDirectory) if
                 os.path.isfile(os.path.join(neuralNetworksDirectory, f))
                 and f.upper()[-4:] == '.PTH']
    neural_network_filepaths.sort(reverse=True)  # Sort in reverse order, such that the worst one is first, the best one is last

    color_coded_images_list = []

    for neural_network_filepath in neural_network_filepaths:
        color_encoded_img = image_to_heatmap_blurs.main(
            imageFilepath=imageFilepath,
            neuralNetworkFilepath=neural_network_filepath,
            device=device,
            architecture=architecture,
            outputDirectory=outputDirectory,
            image_sizeHW=imageSizeHW,
            blurring_sizes=blurringSizes,
            targetHeatmapFilepath=None
        )
        color_encoded_img_filepath = os.path.join(outputDirectory, os.path.basename(neural_network_filepath)[: -4] + '.png')
        cv2.imwrite(color_encoded_img_filepath, color_encoded_img)
        resized_img = cv2.resize(color_encoded_img, (256, 256))
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)  # imageio expects RGB images

        color_coded_images_list.append(resized_img)
    animated_gif_filepath = os.path.join(outputDirectory, "animationMultiNN_" + os.path.basename(imageFilepath)[: -4] + '.gif')
    imageio.mimsave(animated_gif_filepath, color_coded_images_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('neuralNetworksDirectory', help="The directory containing the neural networks")
    parser.add_argument('imageFilepath', help="The filepath to the image")
    parser.add_argument('--useCpu', help='Use cpu, even if cuda is available', action='store_true')
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'FConv_3k5_max_32_64_128'",
                        default='FConv_3k5_max_32_64_128')
    parser.add_argument('--outputDirectory',
                        help="The directory where the output data will be written. Default: './outputs_animationMultiNN'",
                        default='./outputs_animationMultiNN')
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
        args.neuralNetworksDirectory,
        args.imageFilepath,
        device,
        args.architecture,
        args.outputDirectory,
        imageSizeHW,
        blurringSizes
    )