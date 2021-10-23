import logging
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import synthetic_heatmap.generators.stop_sign as stop_sign_gen
import architectures
import ast
import time

parser = argparse.ArgumentParser()
parser.add_argument('validationImagesDirectory', help="The directory that contains the validation images and their corresponding heatmaps")
parser.add_argument('--useCpu', help='Use cpu, even if cuda is available', action='store_true')
parser.add_argument('--architecture', help="The neural network architecture. Default: 'FConv_3k5_max_32_64_128'", default='FConv_3k5_max_32_64_128')
parser.add_argument('--dropoutRatio', help="The dropout ratio. Default: 0.5", type=float, default=0.5)
parser.add_argument('--learningRate', help='The learning rate. Default: 0.0001', type=float, default=0.0001)
parser.add_argument('--weightDecay', help="The weight decay. Default: 0.000001", type=float, default=0.000001)
parser.add_argument('--numberOfEpochs', help='Number of epochs. Default: 1000', type=int, default=1000)
parser.add_argument('--minibatchSize', help='The minibatch size. Default: 16', type=int, default=16)
parser.add_argument('--outputDirectory', help="The directory where the output data will be written. Default: './outputs'", default='./outputs')
parser.add_argument('--imageSizeHW', help="The image size (height, width). Default: '(256, 256)'", default='(256, 256)')
parser.add_argument('--numberOfTrainingSamples', help="The number of training samples generated per epoch. Default: 160", type=int, default=160)
parser.add_argument('--maximumNumberOfValidationSamples', help="The maximum number of validation samples. Default: 160", type=int, default=160)
parser.add_argument('--startingNeuralNetworkFilepath', help="The filepath to the starting neural network. If 'None', starts from randomly initiated weights. Default: 'None'", default='None')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

device = 'cpu'
useCuda = not args.useCpu and torch.cuda.is_available()
if useCuda:
    device = 'cuda'
image_sizeHW = ast.literal_eval(args.imageSizeHW)
startingNeuralNetworkFilepath = None
if args.startingNeuralNetworkFilepath.upper() != 'NONE':
    startingNeuralNetworkFilepath = args.startingNeuralNetworkFilepath

class FixedImageHeatmaps(Dataset):  # For validation, we'll use a fixed dataset
    def __init__(self, image_heatmap_filepath_pairs, maximum_number_of_samples=None, blurring_sizes=[1, 3]):
        super().__init__()
        if maximum_number_of_samples is None:
            self.image_heatmap_filepath_pairs = image_heatmap_filepath_pairs
        else:
            self.image_heatmap_filepath_pairs = image_heatmap_filepath_pairs[:maximum_number_of_samples]
        self.blurring_sizes = blurring_sizes

    def __len__(self):
        return len(self.image_heatmap_filepath_pairs)

    def __getitem__(self, idx):
        input_img_filepath = self.image_heatmap_filepath_pairs[idx][0]
        heatmap_img_filepath = self.image_heatmap_filepath_pairs[idx][1]
        input_img = cv2.imread(input_img_filepath, cv2.IMREAD_GRAYSCALE)
        images = []
        for blurring_size in self.blurring_sizes:
            image = cv2.blur(input_img, (blurring_size, blurring_size))
            images.append(image)
        heatmap_img = cv2.imread(heatmap_img_filepath, cv2.IMREAD_GRAYSCALE)

        input_tsr_list = []
        for image in images:
            input_tsr = torch.from_numpy(image).unsqueeze(0)/256.0  # (1, 256, 256)
            input_tsr_list.append(input_tsr)
        heatmap_tsr = torch.from_numpy(heatmap_img).unsqueeze(0)/256.0  # (1, 256, 256)
        return (input_tsr_list, heatmap_tsr)

class DynamicImageHeatmaps(Dataset):  # For training, we'll generate the data on the fly
    def __init__(self, image_sizeHW, number_of_samples, blurring_sizes=[1, 3]):
        super().__init__()
        self.image_sizeHW = image_sizeHW
        self.number_of_samples = number_of_samples
        self.generator = stop_sign_gen.StopSign()
        self.blurring_sizes = blurring_sizes

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        maximum_number_of_trials = 20
        generation_succeeded = False
        trialNdx = 1
        while not generation_succeeded and trialNdx < maximum_number_of_trials:
            try:
                (input_image, heatmap, result_msg) = self.generator.Generate(self.image_sizeHW)
                generation_succeeded = True
            except:
                logging.warning("DynamicImageHeatmaps.__getitem__(): Generation failed.")
                time.sleep(5.0)
                trialNdx += 1
        images = []
        for blurring_size in self.blurring_sizes:
            image = cv2.blur(input_image, (blurring_size, blurring_size))
            images.append(image)
        input_tsr_list = []
        for image in images:
            input_tsr = torch.from_numpy(image).unsqueeze(0) / 256.0  # (1, 256, 256)
            input_tsr_list.append(input_tsr)

        heatmap_tsr = torch.from_numpy(heatmap).unsqueeze(0)/256.0  # (1, 256, 256)
        return (input_tsr_list, heatmap_tsr)

def ImageHeatmapPairs(directory, image_keyword='input', heatmap_keyword='heatmap'):
    filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if
                 os.path.isfile(os.path.join(directory, f)) ]
    input_img_filepaths = [filepath for filepath in filepaths if image_keyword in os.path.basename(filepath)]
    inputImgHeatmap_filepathPairs = []
    for input_img_filepath in input_img_filepaths:
        filename = os.path.basename(input_img_filepath)
        heatmap_filename = filename.replace(image_keyword, heatmap_keyword)
        heatmap_filepath = os.path.join(directory, heatmap_filename)
        if not os.path.exists(heatmap_filepath):
            raise FileExistsError("ImageHeatmapPairs(): The heatmap filepath {} doesn't exist".format(heatmap_filepath))
        inputImgHeatmap_filepathPairs.append((input_img_filepath, heatmap_filepath))
    return inputImgHeatmap_filepathPairs


def main():
    logging.info("train_stop_sign.py main() useCuda = {}".format(useCuda))

    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    # Create the neural network
    neural_network = None
    if args.architecture == 'FConv_3k5_max_32_64_128':
        neural_network = architectures.FConv_3k5_max(number_of_channels=(32, 64, 128), dropout_ratio=args.dropoutRatio)
    else:
        raise NotImplementedError("train_stop_sign_blursMax.py main(): Not implemented architecture '{}'".format(args.architecture))

    if startingNeuralNetworkFilepath is not None:
        neural_network.load_state_dict(torch.load(startingNeuralNetworkFilepath, map_location=torch.device(device)))
    neural_network = neural_network.to(device)

    # Create the datasets
    validation_dataset = FixedImageHeatmaps(ImageHeatmapPairs(args.validationImagesDirectory), args.maximumNumberOfValidationSamples)
    training_dataset = DynamicImageHeatmaps(image_sizeHW, args.numberOfTrainingSamples)
    # Create the data loaders
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.minibatchSize, shuffle=True, num_workers=2)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.minibatchSize, shuffle=True, num_workers=2)

    # Create the optimizer
    optimizer = torch.optim.Adam(neural_network.parameters(), lr=args.learningRate, betas=(0.5, 0.999),
                                 weight_decay=args.weightDecay)
    # Loss function
    loss_fcn = torch.nn.BCELoss()

    # Output monitoring file
    lowest_validation_loss = 1.0e9
    champion_neural_network = None
    with open(os.path.join(args.outputDirectory, 'epochLoss.csv'), 'w+', buffering=1) as epoch_loss_file:
        epoch_loss_file.write("epoch,training_loss, validation_loss\n")
        for epoch in range(1, args.numberOfEpochs + 1):
            logging.info(" ******* Epoch {} *******".format(epoch))
            # Training
            neural_network.train()
            loss_sum = 0
            number_of_minibatches = 0
            for (input_tsr_list, target_heatmap_tsr) in training_loader:
                print('.', end='', flush=True)
                #for input_tsr in input_tsr_list:  # To do: Move the elements in the list to the device. Requires cuda to test
                #    input_tsr = input_tsr.to(device)
                target_heatmap_tsr = target_heatmap_tsr.to(device)
                optimizer.zero_grad()
                output_tsr = neural_network(input_tsr_list)
                loss = loss_fcn(output_tsr, target_heatmap_tsr)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                number_of_minibatches += 1
            training_loss = loss_sum / number_of_minibatches
            print (' ', end='', flush=True)

            # Validation
            with torch.no_grad():
                neural_network.eval()
                validation_loss_sum = 0
                number_of_minibatches = 0
                for (val_input_tsr_list, val_target_heatmap_tsr) in validation_loader:
                    #for val_input_tsr in val_input_tsr_list:
                    #    val_input_tsr = val_input_tsr.to(device)
                    val_target_heatmap_tsr = val_target_heatmap_tsr.to(device)
                    val_output_tensor = neural_network(val_input_tsr_list)
                    val_loss = loss_fcn(val_output_tensor, val_target_heatmap_tsr)
                    validation_loss_sum += val_loss.item()
                    number_of_minibatches += 1
                validation_loss = validation_loss_sum/number_of_minibatches
                logging.info("training_loss = {};   validation_loss = {}".format(training_loss, validation_loss))
                if validation_loss < lowest_validation_loss:
                    lowest_validation_loss = validation_loss
                    champion_neural_network = neural_network
                if epoch % 20 == 0:
                    neural_network_filepath = os.path.join(args.outputDirectory, args.architecture + "_{:.4f}".format(lowest_validation_loss) + ".pth")
                    torch.save(champion_neural_network.state_dict(), neural_network_filepath)
                epoch_loss_file.write("{},{},{}\n".format(epoch, training_loss, validation_loss))

if __name__ == '__main__':
    main()