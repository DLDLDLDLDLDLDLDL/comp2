from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import numpy as np
from random import shuffle
from utils import backend
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2

def preprocess_input(image):
    # TODO