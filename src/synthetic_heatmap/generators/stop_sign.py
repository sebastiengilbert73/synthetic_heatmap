from synthetic_heatmap.generator import Generator, RegularPolygonVertices, WarpAffinePoints, DownloadRandomImage
import cv2
import numpy as np
import random
import math
import urllib.request
import os

class StopSign(Generator):
    def __init__(self, octogon_diameter_range=(1.3, 1.3),
                 font_scale_range=(0.5, 1.0),
                 graylevel_range=(80, 200),
                 noise_amplitude=10):
                 #number_of_lines_range=(15, 30),
                 #number_of_arcs_range=(15, 30)):

        parameters_dict = {}
        parameters_dict['octogon_diameter_range'] = octogon_diameter_range
        parameters_dict['font_scale_range'] = font_scale_range
        parameters_dict['graylevel_range'] = graylevel_range
        parameters_dict['noise_amplitude'] = noise_amplitude
        #parameters_dict['number_of_lines_range'] = number_of_lines_range
        #parameters_dict['number_of_arcs_range'] = number_of_arcs_range

        super().__init__(parameters_dict)
        self.font_types = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX
                           ]
        #self.background_source_url = "https://picsum.photos"


    def Generate(self, image_sizeHW, maximum_number_of_trials=1000, debug_directory=None,
                 apply_laplacian=False, background_image=None):
        heatmap = np.zeros(image_sizeHW, dtype=np.uint8)
        dilation_kernel = np.ones((3, 3), dtype=np.uint8)

        input_image = background_image
        if background_image is None:
            color_img, result_msg = DownloadRandomImage(image_sizeHW=image_sizeHW)
            input_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

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
        random_octagon_diameter = self.RandomValueInRange('octogon_diameter_range') * 0.2 * min(image_sizeHW[0], image_sizeHW[1])
        random_octagon_center = (round(image_sizeHW[1] * random.random()), round(image_sizeHW[0] * random.random()))
        random_octagon_vertices_arr = RegularPolygonVertices(random_octagon_center, random_octagon_diameter, 8)
        random_octagon_vertices_arr = random_octagon_vertices_arr.astype(int)
        random_octagon_octagon_graylevel = self.RandomValueInRange('graylevel_range')
        #cv2.polylines(input_image, random_octagon_vertices_arr, True, random_octagon_octagon_graylevel)
        cv2.fillPoly(input_image, random_octagon_vertices_arr, random_octagon_octagon_graylevel)
        # Reflections
        if random.random() > 0.5:
            input_image = cv2.flip(input_image, 0)
        if random.random() > 0.5:
            input_image = cv2.flip(input_image, 1)

        # Compute the histogram
        """histogram = cv2.calcHist([input_image], [0], None, [32], [0, 256])
        histogram = [h[0] for h in histogram]
        #print ("Generate(): histogram = {}; len(histogram) = {}".format(histogram, len(histogram)))
        peak_gray_level = 4 + 8 * HistogramPeak(histogram)
        """

        # Write 'STOP'
        text_is_within_limits = False
        trialNdx = 1
        text_bounding_boxYXHW = None
        text_gray_level = np.mean(input_image) + 25 + round(25 * random.random())
        if text_gray_level > 255:
            text_gray_level = 255
        stop_sign_img = np.zeros(image_sizeHW, dtype=np.uint8)
        while not text_is_within_limits and trialNdx <= maximum_number_of_trials:
            center = (round(image_sizeHW[1] * random.random()), round(image_sizeHW[0] * random.random()))
            font_type = random.choice(self.font_types)
            font_scale = self.RandomValueInRange('font_scale_range')
            #text_gray_level = self.RandomValueInRange('graylevel_range')
            text_origin = (round(center[0]), round(center[1]))
            cv2.putText(stop_sign_img, 'STOP', text_origin, font_type, font_scale, text_gray_level)
            #input_image = cv2.dilate(input_image, dilation_kernel)

            # Stretch the text vertically by 1.5 x
            H = image_sizeHW[0]
            alpha = 1.5
            affine_transformation_mtx = np.zeros((2, 3), dtype=float)
            affine_transformation_mtx[0, 0] = 1.0
            affine_transformation_mtx[1, 1] = alpha
            affine_transformation_mtx[1, 2] = H / 2 * (1 - alpha)
            stop_sign_img = cv2.warpAffine(stop_sign_img, affine_transformation_mtx, image_sizeHW)
            # Dilate
            stop_sign_img = cv2.dilate(stop_sign_img, np.ones((3, 3), dtype=np.uint8))
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
        inner_vertices_arr = RegularPolygonVertices(octagon_center, 0.85 * diameter, 8)
        #inner_vertices_arr = inner_vertices_arr.astype(int)
        octagon_graylevel = text_gray_level
        stop_text_img = stop_sign_img.copy()
        cv2.fillPoly(stop_sign_img, vertices_arr, octagon_graylevel//2)
        cv2.polylines(stop_sign_img, vertices_arr, True, octagon_graylevel, thickness=3)
        # Re-write the text
        stop_sign_img = cv2.max(stop_sign_img, stop_text_img)

        # Add noise
        stop_sign_img += (self.parameters_dict['noise_amplitude'] * np.random.random(stop_sign_img.shape)).astype(np.uint8)

        # Blur the stop sign, such that its edges are not too sharp
        if random.random() > 0.5:
            stop_sign_img = cv2.blur(stop_sign_img, (3, 3))

        # Erase the input image in the octagon, such that the stop sign is on the foreground
        cv2.fillPoly(input_image, vertices_arr, 0)


        input_image = cv2.max(input_image, stop_sign_img)


        # Affine transformation
        linear_transformation_mtx = np.random.uniform(0.6, 1.5, (2, 2))
        linear_transformation_mtx[0, 1] = -0.3 + 0.6 * random.random()
        linear_transformation_mtx[1, 0] = -0.3 + 0.6 * random.random()
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
        # Check if the heatmap is non-empty
        number_of_non_zero_pixels = cv2.countNonZero(heatmap)
        if diameter > 0:
            circle_filling_ratio = number_of_non_zero_pixels / (math.pi * (diameter/2)**2)
        else:
            circle_filling_ratio = 0
        #print ("number_of_non_zero_pixels = {}; circle_filling_ratio = {}".format(number_of_non_zero_pixels, circle_filling_ratio))
        if circle_filling_ratio < 0.5:
            # The stop sign is not visible enough. Start again
            return self.Generate(image_sizeHW, maximum_number_of_trials, debug_directory)

        if debug_directory is not None:
            input_image_before_laplacian_filepath = os.path.join(debug_directory, "stopSign_Generate_beforeLaplacian.png")
            cv2.imwrite(input_image_before_laplacian_filepath, input_image)
        if apply_laplacian:
            # Compute the edges of the input image
            input_image = cv2.Laplacian(input_image, ddepth=cv2.CV_8U)

        return (input_image, heatmap, result_msg)

def HistogramPeak(histogram):
    peak_value = 0
    peakNdx = None
    for index in range(len(histogram)):
        if histogram[index] > peak_value:
            peak_value = histogram[index]
            peakNdx = index
    return peakNdx
