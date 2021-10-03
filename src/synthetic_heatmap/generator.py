import abc
import random
import math
import numpy as np

class Generator(abc.ABC):
    def __init__(self, parameters_dict):
        self.parameters_dict = parameters_dict

    @abc.abstractmethod
    def Generate(self, image_sizeHW, maximum_number_of_trials):
        pass  # return (input_image, heatmap)

    def RandomValueInRange(self, parameter_name, must_be_rounded=False):
        if not parameter_name in self.parameters_dict:
            raise KeyError("Generator.RandomValueInRange(): Parameter name '{}' not found in the parameters dictionary")
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

#def WarpAffinePoints():