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

class CharString(Generator):
    def __init__(self,
                 font_scale_range=(0.2, 2.0),
                 graylevel_range=(10, 240),
                 noise_amplitude_range=(2, 20),
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

        # Generate random text
        number_of_strings = self.RandomValueInRange('number_of_strings_range', must_be_rounded=True)
        text_img = np.zeros(image_sizeHW, dtype=np.uint8)
        for stringNdx in range(number_of_strings):
            number_of_characters = self.RandomValueInRange('number_of_characters_range', must_be_rounded=True)
            random_string = ''.join(random.choice(self.characters) for i in range(number_of_characters))
            bottom_left = (random.randint(0, image_sizeHW[1] - 1), random.randint(0, image_sizeHW[0] - 1))
            font = random.choice(self.font_types)
            scale = self.RandomValueInRange('font_scale_range')
            logging.debug("CharString.Generate(): scale = {}".format(scale))
            thickness = 1
            if scale > 1:
                thickness = round(scale)
            graylevel = self.RandomValueInRange('graylevel_range', must_be_rounded=True)
            cv2.putText(text_img, random_string, bottom_left, font, scale, graylevel, thickness)

        if debug_directory is not None:
            cv2.imwrite(os.path.join(debug_directory, "CharString_Generate_text.png"), text_img)

        return generated_img, heatmap