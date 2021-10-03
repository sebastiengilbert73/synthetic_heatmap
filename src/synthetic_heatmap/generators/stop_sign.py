from synthetic_heatmap.generator import Generator, RegularPolygonVertices
import cv2
import numpy as np
import random
import math

class StopSign(Generator):
    def __init__(self, octogon_diameter_range=(1.4, 1.6), font_scale_range=(0.5, 2.0),
                 graylevel_range=(40, 255)):
        parameters_dict = {}
        parameters_dict['octogon_diameter_range'] = octogon_diameter_range
        parameters_dict['font_scale_range'] = font_scale_range
        parameters_dict['graylevel_range'] = graylevel_range

        super().__init__(parameters_dict)
        self.font_types = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
                           cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL
                           ]


    def Generate(self, image_sizeHW, maximum_number_of_trials=1000):
        input_image = np.zeros(image_sizeHW, dtype=np.uint8)
        heatmap = np.zeros(image_sizeHW, dtype=np.uint8)
        center = (round(image_sizeHW[1] * random.random()), round(image_sizeHW[0] * random.random()))


        # Write 'STOP'
        text_is_within_limits = False
        trialNdx = 1
        dilation_kernel = np.ones((3, 3), dtype=np.uint8)
        while not text_is_within_limits and trialNdx <= maximum_number_of_trials:
            font_type = random.choice(self.font_types)
            if random.random() > 0.5:
                font_type = font_type + cv2.FONT_ITALIC
            font_scale = self.RandomValueInRange('font_scale_range')
            text_gray_level = self.RandomValueInRange('graylevel_range')
            text_origin = (round(center[0] ), round(center[1] ))
            cv2.putText(input_image, 'STOP', text_origin, font_type, font_scale, text_gray_level)
            input_image = cv2.dilate(input_image, dilation_kernel)

            # Find the bounding box around the text
            non_zero_points = np.transpose(np.nonzero(input_image))
            text_bounding_boxYXHW = cv2.boundingRect(np.array(non_zero_points))
            if text_bounding_boxYXHW[0] + text_bounding_boxYXHW[2] < image_sizeHW[0] - 8 and \
                    text_bounding_boxYXHW[1] + text_bounding_boxYXHW[3] < image_sizeHW[1] - 8:
                text_is_within_limits = True
                print("text_bounding_boxYXHW = {}".format(text_bounding_boxYXHW))
            else:  # The text touches the limit
                text_is_within_limits = False
                input_image = np.zeros(image_sizeHW, dtype=np.uint8)


        diameter = self.RandomValueInRange('octogon_diameter_range', True) * text_bounding_boxYXHW[3]
        octagon_center = (text_bounding_boxYXHW[1] + text_bounding_boxYXHW[3]/2, text_bounding_boxYXHW[0] + text_bounding_boxYXHW[2]/2)
        vertices_arr = RegularPolygonVertices(octagon_center, diameter, 8)
        vertices_arr = vertices_arr.astype(int)
        octagon_graylevel = self.RandomValueInRange('graylevel_range')
        cv2.polylines(input_image, vertices_arr, True, octagon_graylevel)



        return (input_image, heatmap)


