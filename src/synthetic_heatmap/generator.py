import abc
import random
import math
import numpy as np
import urllib
import cv2

class Generator(abc.ABC):
    def __init__(self, parameters_dict):
        self.parameters_dict = parameters_dict

    @abc.abstractmethod
    def Generate(self, image_sizeHW, maximum_number_of_trials):
        pass  # return (input_image, heatmap)

    def RandomValueInRange(self, parameter_name, must_be_rounded=False):
        if not parameter_name in self.parameters_dict:
            raise KeyError("Generator.RandomValueInRange(): Parameter name '{}' not found in the parameters dictionary".format(parameter_name))
        parameter_range = self.parameters_dict[parameter_name]
        random_value = parameter_range[0] + (parameter_range[1] - parameter_range[0]) * random.random()
        if must_be_rounded:
            return round(random_value)
        else:
            return random_value

def RegularPolygonVertices(center, diameter, number_of_vertices):
    radius = diameter/2
    vertices_list = []
    for thetaNdx in range(number_of_vertices):
        theta = 2 * math.pi * thetaNdx/number_of_vertices + math.pi/number_of_vertices
        vertex = [center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)]
        vertices_list.append(vertex)
    return np.array([vertices_list])

def WarpAffinePoints(points_arr, affine_transformation_mtx):  # points_arr.shape = (1, N, 2) The 1st dimension is the number of polylines (assuming 1)
    warped_points = np.zeros((1, points_arr.shape[1], 2))
    for ptNdx in range(points_arr.shape[1]):
        original_pt = np.array([points_arr[0, ptNdx, 0], points_arr[0, ptNdx, 1], 1])
        transformed_pt = affine_transformation_mtx @ original_pt
        warped_points[0, ptNdx, 0] = transformed_pt[0]
        warped_points[0, ptNdx, 1] = transformed_pt[1]
    return warped_points.astype(int)

def DownloadRandomImage(method='picsum', image_sizeHW=(256, 256)):
    if method == 'picsum':
        temp_filepath = "./stock-image.jpg"
        url = "https://picsum.photos"
        trialNdx = 1
        download_was_successful = False
        image = None
        result_msg = None
        while trialNdx < 20 and not download_was_successful:
            (data_file, result_msg) = urllib.request.urlretrieve(
                url + '/' + str(image_sizeHW[1]) + '/' + str(image_sizeHW[0]),
                temp_filepath)  # Inspired from https://gist.github.com/jeff-mccarthy/3e04e39014046ffb36e36e001da74449
            image = cv2.imread(temp_filepath)

            trialNdx = 1
            download_was_successful = True
            # Test if the download was successful
            if image is None:
                trialNdx += 1
                download_was_successful = False
            """try:
                input_image = cv2.max(image, image)
            except Exception as ex:
                trialNdx += 1
                download_was_successful = False
            """
        if trialNdx >= 20:
            raise ValueError("DownloadRandomImage(): 20 trials to download an image was reached")
        return image, result_msg
    else:
        raise NotImplementedError("DownloadRandomImage(): Not implemented method '{}'".format(method))