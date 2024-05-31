
from DataExtractionandResizing import vtu_to_numpy, resize
from Interpolation import interpolate, extractWingData

from ml_collections import ConfigDict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def vtu2structured(config: ConfigDict, vtu_file: str, stl_file: str, mach: float):
	
	#xmin = -0.75
	#xmax = 1.25
	#ymin = -1
	#ymax = 1



	x = vtu_to_numpy(vtu_file)

	xmin, xmax, ymin, ymax = config.preprocess.dim
	data = resize(x, xmin, xmax, ymin, ymax)



	wing_data = extractWingData(stl_file)




#	mach = float(vtu_file[vtu_file.rfind("_")+1:-4])
	x, y = interpolate(config, data, wing_data, mach)


	w, h = config.vit.img_size
	c_encoder, c_decoder = x.shape[1], y.shape[1]
	x = x.reshape([w, h, c_encoder])
	y = y.reshape([w, h, c_decoder])

	return x, y




def create_tfExample(x: np.ndarray, y: np.ndarray, config: str):
    """ Create example for data sample which will store the necessary
    information about each CFD simulation

    Parameters
    ----------
    y: np.ndarray
            encoder input
    x: np.ndarray
            decoder input
    config: string
            simulation configuration with airfoil, aoa, mach number

    Returns
    -------
    example: tensorflow.train.Example
             returns data in the Example format to write into TFRecord
    """

    feature = {
        'encoder': tf.train.Feature(float_list=tf.train.FloatList(
            value=x.flatten())),
        'decoder': tf.train.Feature(float_list=tf.train.FloatList(
            value=y.flatten())),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[config.encode()]))
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))
