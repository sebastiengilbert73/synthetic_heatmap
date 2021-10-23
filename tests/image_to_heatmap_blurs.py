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
        device,
        architecture,
        outputDirectory,
        image_sizeHW,
        blurring_sizes,
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

    input_imgs = []
    input_tsrs = []
    for blurring_size in blurring_sizes:
        input_img = cv2.blur(resized_img, (blurring_size, blurring_size))
        input_imgs.append(input_img)
        input_img_filepath = os.path.join(outputDirectory, "imageToHeatmapBlurs_main_inputBlur{}.png".format(blurring_size))
        cv2.imwrite(input_img_filepath, input_img)

        # Conversion to a tensor
        input_tsr = torch.from_numpy(input_img).unsqueeze(0)/256.0  # [1, 256, 256]
        input_tsr = input_tsr.to(device)
        input_tsrs.append(input_tsr.unsqueeze(0))

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
    elif architecture == 'FConv_3k5_max_32_64_128':
        neural_network = architectures.FConv_3k5_max(number_of_channels=(32, 64, 128))
    else:
        raise NotImplementedError("main(): Not implemented architecture '{}'".format(architecture))
    neural_network.load_state_dict(torch.load(neuralNetworkFilepath, map_location=torch.device(device)))
    neural_network = neural_network.to(device)
    neural_network.eval()

    # Pass the input tensors through the neural network
    output_tsrs = [neural_network(input_tsr.unsqueeze(0)) for input_tsr in input_tsrs]
    output_tsr = output_tsrs[0]
    for additional_output_tsr_ndx in range(1, len(output_tsrs)):
        output_tsr = torch.max(output_tsr, output_tsrs[additional_output_tsr_ndx])

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
        target_output_tsr = target_heatmap_tsr.unsqueeze(0)
        lossFcn = torch.nn.BCELoss()
        loss = lossFcn(output_tsr, target_output_tsr.detach())
        logging.info("main(): loss = {}".format(loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imageFilepath', help="The image filepath")
    parser.add_argument('neuralNetworkFilepath', help="The filepath to the neural network")
    parser.add_argument('--useCpu', help='Use cpu, even if cuda is available', action='store_true')
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'FConv_4k5_16_32_64_128'",
                        default='FConv_4k5_16_32_64_128')
    parser.add_argument('--outputDirectory',
                        help="The directory where the output data will be written. Default: './outputs'",
                        default='./outputs')
    parser.add_argument('--imageSizeHW', help="The image size (height, width). Default: '(256, 256)'",
                        default='(256, 256)')
    parser.add_argument('--blurringSizes', help="The blurring sizes. Default: '[1, 3]'", default='[1, 3]')
    parser.add_argument('--targetHeatmapFilepath', help="If the target heatmap is passed, the loss will be computed. Default: 'None'", default='None')
    args = parser.parse_args()

    device = 'cpu'
    useCuda = not args.useCpu and torch.cuda.is_available()
    if useCuda:
        device = 'cuda'
    image_sizeHW = ast.literal_eval(args.imageSizeHW)
    blurring_sizes = ast.literal_eval(args.blurringSizes)
    if args.targetHeatmapFilepath.upper() == 'NONE':
        targetHeatmapFilepath = None
    else:
        targetHeatmapFilepath = args.targetHeatmapFilepath

    main(
        args.imageFilepath,
        args.neuralNetworkFilepath,
        device,
        args.architecture,
        args.outputDirectory,
        image_sizeHW,
        blurring_sizes,
        targetHeatmapFilepath
    )