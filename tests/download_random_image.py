import os
import argparse
import logging
import cv2
import numpy as np
from  synthetic_heatmap.generator import DownloadRandomImage

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')

def main():
    logging.info("download_random_image.main()")

    image, result_msg = DownloadRandomImage()
    cv2.imwrite("./outputs/downloadRandomImage_image.png", image)


if __name__ == '__main__':
    main()