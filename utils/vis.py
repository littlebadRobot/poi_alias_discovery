import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from pyheatmap.heatmap import HeatMap

def vis_heat_map(data):
    plt.matshow(data, cmap='afmhot')
    plt.colorbar()
