import copy

from synthetic_heatmap.generator import Generator, DownloadRandomImage
import cv2
import numpy as np
import random
import math
import os
import copy
import string
import logging
import imutils

class CharString(Generator):
    def __init__(self,
                 font_scale_range=(0.5, 2.0),
                 graylevel_range=(30, 100),
                 noise_amplitude_range=(1, 3),
                 number_of_characters_range=(2, 20),
                 number_of_strings_range=(1, 5),
                 font_types=[cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
                             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX
                           ],
                 characters=string.ascii_letters + string.digits):

        parameters_dict = {}
        parameters_dict['font_scale_range'] = font_scale_range
        parameters_dict['graylevel_range'] = graylevel_range
        parameters_dict['noise_amplitude_range'] = noise_amplitude_range
        parameters_dict['number_of_characters_range'] = number_of_characters_range
        parameters_dict['number_of_strings_range'] = number_of_strings_range
        super().__init__(parameters_dict)
        self.font_types = font_types
        self.characters = characters
        self.rng = np.random.default_rng()

    def Generate(self, image_sizeHW,
                 maximum_number_of_trials=None,
                 debug_directory=None,
                 background_image=None):
        if background_image is None:
            background_image, result_msg = DownloadRandomImage(method='picsum', image_sizeHW=image_sizeHW)
        else:
            if background_image.shape[0] != image_sizeHW[0] or background_image.shape[1] != image_sizeHW[1]:
                background_image = cv2.resize(background_image, image_sizeHW)

        # Convert to grayscale
        grayscale_img = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
        generated_img = copy.deepcopy(grayscale_img)
        heatmap = np.zeros(image_sizeHW, dtype=np.uint8)
        average_graylevel = np.mean(grayscale_img)

        # Generate random text
        number_of_strings = self.RandomValueInRange('number_of_strings_range', must_be_rounded=True)
        text_img = np.zeros(image_sizeHW, dtype=np.uint8)
        for stringNdx in range(number_of_strings):
            number_of_characters = self.RandomValueInRange('number_of_characters_range', must_be_rounded=True)
            random_string = ''.join(random.choice(self.characters) for i in range(number_of_characters))
            bottom_left = (random.randint(0, image_sizeHW[1] - 1), random.randint(0, image_sizeHW[0] - 1))
            font = random.choice(self.font_types)
            scale = self.RandomValueInRange('font_scale_range')
            #logging.debug("CharString.Generate(): scale = {}".format(scale))
            thickness = 2
            if scale > 1:
                thickness = 1 + round(scale)
            graylevel = self.RandomValueInRange('graylevel_range', must_be_rounded=True)
            if np.random.randn() < 0.5:
                graylevel = int( np.clip( round(average_graylevel - graylevel), 0, 255))
            else:
                graylevel = int( np.clip( round(average_graylevel + graylevel), 0, 255))

            cv2.putText(text_img, random_string, bottom_left, font, scale, graylevel, thickness)

        # Random rotation
        theta_in_degrees = 360 * np.random.rand()
        text_img = imutils.rotate(text_img, theta_in_degrees)

        # The eroded text mask
        _, eroded_text_mask = cv2.threshold(text_img, self.parameters_dict['graylevel_range'][0], 255, cv2.THRESH_BINARY)
        text_erosion_kernel = np.ones((1, 1), dtype=np.uint8)
        eroded_text_mask = cv2.erode(eroded_text_mask, text_erosion_kernel)

        # Heatmap: threshold the text image
        _, heatmap = cv2.threshold(text_img, self.parameters_dict['graylevel_range'][0] , 255, cv2.THRESH_BINARY)
        dilation_kernel = np.ones((5, 5), dtype=np.uint8)
        heatmap = cv2.dilate(heatmap, dilation_kernel)

        boundary_kernel = np.ones((5, 5), dtype=np.uint8)
        boundary_mask = cv2.dilate(heatmap, boundary_kernel) - heatmap

        # Generate noise
        noise_sigma = self.RandomValueInRange('noise_amplitude_range')
        noise_img = noise_sigma * self.rng.standard_normal(image_sizeHW)

        # Blur the background image, as the text background must be approximately uniform
        blurring_size = (11, 11)
        blurred_background_img = cv2.blur(grayscale_img, blurring_size)
        generated_img = (np.where(heatmap > 0, blurred_background_img + noise_img, grayscale_img)).astype(np.uint8)

        blurred_generated_img = cv2.blur(generated_img, (3, 3))
        # Scan the heatmap
        for y in range(heatmap.shape[0]):
            for x in range(heatmap.shape[1]):
                if eroded_text_mask[y, x] > 0:
                    generated_img[y, x] = text_img[y, x] + noise_img[y, x]
                elif boundary_mask[y, x] > 0:
                    generated_img[y, x] = blurred_generated_img[y, x] + noise_img[y, x]

        blurred_generated_img = cv2.blur(generated_img, (3, 3))
        generated_img = np.where(heatmap , blurred_generated_img, generated_img)

        if debug_directory is not None:
            cv2.imwrite(os.path.join(debug_directory, "CharString_Generate_text.png"), text_img)

        return generated_img, heatmap