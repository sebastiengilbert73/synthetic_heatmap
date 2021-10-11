import logging
import argparse
import torch
import os
import cv2
import architectures
import ast

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def main(
        imageFilepath,
        neuralNetworkFilepath,
        applyLaplacian,
        device,
        architecture,
        outputDirectory,
        image_sizeHW,
        blurringSize,
        targetHeatmapFilepath
    ):
    logging.info("image_to_heatmap.py main(); device = {}".format(device))

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    original_img = cv2.imread(imageFilepath, cv2.IMREAD_GRAYSCALE)

    # Resizing
    resized_img = None
    if original_img.shape != image_sizeHW:
        logging.info("main(): Resizing to {}".format(image_sizeHW))
        resized_img = cv2.resize(original_img, (image_sizeHW[1], image_sizeHW[0]))
    else:
        resized_img = original_img

    # Edge detection
    edges_img = None
    if applyLaplacian:
        logging.info("main(): Applying Laplacian")
        edges_img = cv2.Laplacian(resized_img, ddepth=cv2.CV_8U)
    else:
        edges_img = resized_img
    edges_img = cv2.blur(edges_img, (blurringSize, blurringSize))
    edges_img_filepath = os.path.join(outputDirectory, "imageToHeatmap_main_edges.png")
    cv2.imwrite(edges_img_filepath, edges_img)

    # Conversion to a tensor
    edges_tsr = torch.from_numpy(edges_img).unsqueeze(0)/256.0  # [1, 256, 256]
    edges_tsr = edges_tsr.to(device)

    # Neural network loading
    neural_network = None
    if architecture == 'FConv_3k5_32_64_128':
        neural_network = architectures.FConv_3k5(number_of_channels=(32, 64, 128))
    elif architecture == 'FConv_4k5_16_32_64_128':
        neural_network = architectures.FConv_4k5(number_of_channels=(16, 32, 64, 128))
    elif architecture == 'FConv_4k5_32_64_128_256':
        neural_network = architectures.FConv_4k5(number_of_channels=(32, 64, 128, 256))
    elif architecture == 'FConv_4k5_64_128_256_512':
        neural_network = architectures.FConv_4k5(number_of_channels=(64, 128, 256, 512))
    elif architecture == 'FConv_5k5_16_32_64_128_256':
        neural_network = architectures.FConv_5k5(number_of_channels=(16, 32, 64, 128, 256))
    else:
        raise NotImplementedError("main(): Not implemented architecture '{}'".format(architecture))
    neural_network.load_state_dict(torch.load(neuralNetworkFilepath, map_location=torch.device(device)))
    neural_network = neural_network.to(device)
    neural_network.eval()

    # Pass the input tensor through the neural network
    output_tsr = neural_network(edges_tsr.unsqueeze(0))

    # Convert to an image
    output_img = (256 * output_tsr).squeeze().detach().numpy()
    output_img_filepath = os.path.join(outputDirectory, "imageToHeatmap_main_output.png")
    cv2.imwrite(output_img_filepath, output_img)

    # Resize to the original image
    output_heatmap = output_img
    if output_heatmap.shape != original_img.shape:
        logging.info("Resizing to {}".format(original_img.shape))
        output_heatmap = cv2.resize(output_heatmap, (original_img.shape[1], original_img.shape[0]))
    output_heatmap_filepath = os.path.join(outputDirectory, "imageToHeatmap_main_heatmap.png")
    cv2.imwrite(output_heatmap_filepath, output_heatmap)

    if targetHeatmapFilepath is not None:
        target_heatmap = cv2.imread(targetHeatmapFilepath, cv2.IMREAD_GRAYSCALE)
        target_heatmap_tsr = torch.from_numpy(target_heatmap).unsqueeze(0)/256.0
        target_heatmap_tsr = target_heatmap_tsr.to(device)
        target_output_tsr = neural_network(target_heatmap_tsr.unsqueeze(0))
        lossFcn = torch.nn.BCELoss()
        loss = lossFcn(output_tsr, target_output_tsr.detach())
        logging.info("main(): loss = {}".format(loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imageFilepath', help="The image filepath")
    parser.add_argument('neuralNetworkFilepath', help="The filepath to the neural network")
    parser.add_argument('--applyLaplacian', help="Flag to apply a Laplacian edge detector", action='store_true')
    parser.add_argument('--useCpu', help='Use cpu, even if cuda is available', action='store_true')
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'FConv_4k5_16_32_64_128'",
                        default='FConv_4k5_16_32_64_128')
    parser.add_argument('--outputDirectory',
                        help="The directory where the output data will be written. Default: './outputs'",
                        default='./outputs')
    parser.add_argument('--imageSizeHW', help="The image size (height, width). Default: '(256, 256)'",
                        default='(256, 256)')
    parser.add_argument('--blurringSize', help="The blurring size. Default: 3", type=int, default=3)
    parser.add_argument('--targetHeatmapFilepath', help="If the target heatmap is passed, the loss will be computed. Default: 'None'", default='None')
    args = parser.parse_args()

    device = 'cpu'
    useCuda = not args.useCpu and torch.cuda.is_available()
    if useCuda:
        device = 'cuda'
    image_sizeHW = ast.literal_eval(args.imageSizeHW)
    if args.targetHeatmapFilepath.upper() == 'NONE':
        targetHeatmapFilepath = None
    else:
        targetHeatmapFilepath = args.targetHeatmapFilepath

    main(
        args.imageFilepath,
        args.neuralNetworkFilepath,
        args.applyLaplacian,
        device,
        args.architecture,
        args.outputDirectory,
        image_sizeHW,
        args.blurringSize,
        targetHeatmapFilepath
    )