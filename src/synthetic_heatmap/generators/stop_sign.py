from synthetic_heatmap.generator import Generator, RegularPolygonVertices, WarpAffinePoints
import cv2
import numpy as np
import random
import math

class StopSign(Generator):
    def __init__(self, octogon_diameter_range=(1.3, 1.5),
                 font_scale_range=(0.5, 1.0),
                 graylevel_range=(40, 200),
                 number_of_lines_range=(15, 30),
                 number_of_arcs_range=(15, 30)):

        parameters_dict = {}
        parameters_dict['octogon_diameter_range'] = octogon_diameter_range
        parameters_dict['font_scale_range'] = font_scale_range
        parameters_dict['graylevel_range'] = graylevel_range
        parameters_dict['number_of_lines_range'] = number_of_lines_range
        parameters_dict['number_of_arcs_range'] = number_of_arcs_range

        super().__init__(parameters_dict)
        self.font_types = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
                           cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL
                           ]


    def Generate(self, image_sizeHW, maximum_number_of_trials=1000):
        input_image = np.zeros(image_sizeHW, dtype=np.uint8)
        heatmap = np.zeros(image_sizeHW, dtype=np.uint8)
        dilation_kernel = np.ones((3, 3), dtype=np.uint8)
        blurring_size = 3



        # Draw random lines and curves
        number_of_lines = self.RandomValueInRange('number_of_lines_range', True)
        for lineNdx in range(number_of_lines):
            line_graylevel = self.RandomValueInRange('graylevel_range', True)
            p1 = (round (image_sizeHW[1] * random.random()), round(image_sizeHW[0] * random.random()))
            p2 = (round (image_sizeHW[1] * random.random()), round(image_sizeHW[0] * random.random()))
            cv2.line(input_image, p1, p2, line_graylevel)

        number_of_arcs = self.RandomValueInRange('number_of_lines_range', True)
        for arcNdx in range(number_of_arcs):
            ellipse_center = (round(image_sizeHW[1] * random.random()), round(image_sizeHW[0] * random.random()))
            a = round(max(image_sizeHW[0], image_sizeHW[1]) * random.random())
            b = round(max(image_sizeHW[0], image_sizeHW[1]) * random.random())
            ellipse_graylevel = self.RandomValueInRange('graylevel_range', True)
            cv2.ellipse(input_image, ellipse_center, (a, b), round(360 * random.random()), round(360 * random.random()) ,
                        round(360 * random.random()), ellipse_graylevel)

        # Write a random 'STOP'
        random_stop_img = np.zeros(image_sizeHW, dtype=np.uint8)
        random_stop_center = (round(image_sizeHW[1] * random.random()), round(image_sizeHW[0] * random.random()))
        font_type = random.choice(self.font_types)
        font_scale = self.RandomValueInRange('font_scale_range')
        text_gray_level = self.RandomValueInRange('graylevel_range')
        text_origin = (round(random_stop_center[0]), round(random_stop_center[1]))
        cv2.putText(random_stop_img, 'STOP', text_origin, font_type, font_scale, text_gray_level)
        random_stop_img = cv2.dilate(random_stop_img, dilation_kernel)
        input_image = cv2.max(input_image, random_stop_img)

        # Draw a random octagon
        random_octagon_diameter = self.RandomValueInRange('octogon_diameter_range') * math.sqrt(
            0.2 * min(image_sizeHW[0], image_sizeHW[1]))
        random_octagon_center = (round(image_sizeHW[1] * random.random()), round(image_sizeHW[0] * random.random()))
        random_octagon_vertices_arr = RegularPolygonVertices(random_octagon_center, random_octagon_diameter, 8)
        random_octagon_vertices_arr = random_octagon_vertices_arr.astype(int)
        random_octagon_octagon_graylevel = self.RandomValueInRange('graylevel_range')
        cv2.polylines(input_image, random_octagon_vertices_arr, True, random_octagon_octagon_graylevel)

        # Write 'STOP'
        text_is_within_limits = False
        trialNdx = 1
        text_bounding_boxYXHW = None
        text_gray_level = None
        stop_sign_img = np.zeros(image_sizeHW, dtype=np.uint8)
        while not text_is_within_limits and trialNdx <= maximum_number_of_trials:
            center = (round(image_sizeHW[1] * random.random()), round(image_sizeHW[0] * random.random()))
            font_type = random.choice(self.font_types)
            font_scale = self.RandomValueInRange('font_scale_range')
            text_gray_level = self.RandomValueInRange('graylevel_range')
            text_origin = (round(center[0]), round(center[1]))
            cv2.putText(stop_sign_img, 'STOP', text_origin, font_type, font_scale, text_gray_level)
            input_image = cv2.dilate(input_image, dilation_kernel)

            # Find the bounding box around the text
            non_zero_points = np.transpose(np.nonzero(stop_sign_img))
            text_bounding_boxYXHW = cv2.boundingRect(np.array(non_zero_points))
            if text_bounding_boxYXHW[0] + text_bounding_boxYXHW[2] < image_sizeHW[0] - 8 and \
                    text_bounding_boxYXHW[1] + text_bounding_boxYXHW[3] < image_sizeHW[1] - 8:
                text_is_within_limits = True
            else:  # The text touches the limit
                text_is_within_limits = False
                stop_sign_img = np.zeros(image_sizeHW, dtype=np.uint8)
            trialNdx += 1

        diameter = self.RandomValueInRange('octogon_diameter_range') * math.sqrt(
            text_bounding_boxYXHW[3] ** 2 + text_bounding_boxYXHW[2] ** 2)
        octagon_center = (text_bounding_boxYXHW[1] + text_bounding_boxYXHW[3] / 2,
                          text_bounding_boxYXHW[0] + text_bounding_boxYXHW[2] / 2)
        vertices_arr = RegularPolygonVertices(octagon_center, diameter, 8)
        vertices_arr = vertices_arr.astype(int)
        octagon_graylevel = self.RandomValueInRange('graylevel_range')
        cv2.polylines(stop_sign_img, vertices_arr, True, octagon_graylevel)
        # Erase the input image in the octagon, such that the stop sign is on the foreground
        cv2.fillPoly(input_image, vertices_arr, 0)
        input_image = cv2.max(input_image, stop_sign_img)


        # Affine transformation
        linear_transformation_mtx = np.random.uniform(0.6, 1.5, (2, 2))
        linear_transformation_mtx[0, 1] = 0.3 * random.random()
        linear_transformation_mtx[1, 0] = 0.3 * random.random()
        translation = [image_sizeHW[1]/2 - (linear_transformation_mtx @ np.array([image_sizeHW[1]/2, image_sizeHW[0]/2]))[0],
                       image_sizeHW[0] / 2 - (linear_transformation_mtx @ np.array(
                           [image_sizeHW[1] / 2, image_sizeHW[0] / 2] ))[1] ]
        affine_transformation_mtx = np.zeros((2, 3))
        affine_transformation_mtx[0: 2, 0: 2] = linear_transformation_mtx
        affine_transformation_mtx[0, 2] = translation[0]
        affine_transformation_mtx[1, 2] = translation[1]
        input_image = cv2.warpAffine(input_image, affine_transformation_mtx, input_image.shape)

        affine_transformed_vertices = WarpAffinePoints(vertices_arr, affine_transformation_mtx)
        cv2.fillPoly(heatmap, affine_transformed_vertices, 255)

        # Compute the edges of the input image
        input_image = cv2.Laplacian(input_image, ddepth=cv2.CV_8U)
        input_image = cv2.blur(input_image, (blurring_size, blurring_size))

        return (input_image, heatmap)


