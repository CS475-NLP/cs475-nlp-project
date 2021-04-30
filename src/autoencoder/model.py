import random
import matplotlib.pyplot as plt
import pickle as pkl
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, multiply, concatenate, Flatten, Activation, dot
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model, model_to_dot
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
