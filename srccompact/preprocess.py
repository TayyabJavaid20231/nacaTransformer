
from ml_collections import ConfigDict
from conversion import vtu2structured, create_tfExample
from Utilities import prefilter_dataset


import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


def generate_tfds_dataset(config: ConfigDict):
    """
    Convert vtk datasets from airfoilMNIST (config.preprocess.readdir) into the TFRecord format
    and save them as into the TFRecords directory (writedir). More
    information about TFRecords can be found on
    https://www.tensorflow.org/tutorials/load_data/tfrecord.
    """

    fileRead = open('VTUFiles.txt', 'r')
    files = fileRead.readlines()
    fileRead.close
    files = [x[:-1] for x in files]


  
# check if output directory exists and create dir if necessary
    if not os.path.exists(config.preprocess.writedir):
        os.makedirs(config.preprocess.writedir)

    vtu_folder = os.path.join(config.preprocess.readdir, 'vtu')
    stl_folder = os.path.join(config.preprocess.readdir, 'stl')
    


#change here
    vtu_list = [x for x in sorted(files) if x.endswith('.vtu')]
    vtu_list = prefilter_dataset(vtu_list, aoa_limits=config.preprocess.aoa,
                                 mach_limits=config.preprocess.mach)

    

    stl_list = [("_".join(x.split("_", 2)[:2]) + ".stl") for x in vtu_list]

    dataset = list(zip(vtu_list, stl_list))
#change here
    ds_train = [dataset.pop(random.randrange(len(dataset))) for _ in
                range(int(config.preprocess.train_split * len(dataset)))]
    ds_test = dataset

    train_quotient, train_remainder = divmod(len(ds_train),
                                             config.preprocess.nsamples)
    test_quotient, test_remainder = divmod(len(ds_test),
                                           config.preprocess.nsamples)

    n_files_train = (train_quotient + 1 if train_remainder != 0 else
                     train_quotient)
    n_files_test = (test_quotient + 1 if test_remainder != 0 else test_quotient)

    train_shards, test_shards = [], []

    for i in tqdm(range(n_files_train), desc='Train split', position=0):
        if train_remainder != 0 and i == n_files_train - 1:
            batch = [ds_train.pop(random.randrange(len(ds_train))) for _ in
                     range(train_remainder)]
        else:
            batch = [ds_train.pop(random.randrange(len(ds_train))) for _ in
                     range(config.preprocess.nsamples)]

        file_dir = os.path.join(config.preprocess.writedir,
                                'airfoilMNIST-train.tfrecord-{}-of-{}'.format(
                                    str(i).zfill(5),
                                    str(n_files_train).zfill(
                                        5)))

        with tf.io.TFRecordWriter(file_dir) as writer:
            j = 0
            for sample in tqdm(batch, desc='Shards', position=1,
                               leave=False):
                # split string to extract information about airfoil, angle of
                # attack and Mach number to write as feature into tfrecord
                sim_config = sample[0].rsplit('.', 1)[0]

                airfoil, angle, mach = sim_config.split('_')
                angle, mach = float(angle), float(mach)
                

#change here
                #copyfiles(sample[0], sample[1])
                vtu_dir = os.path.join(vtu_folder, sample[0])
                stl_dir = os.path.join(stl_folder, sample[1])
                try:
                    x, y = vtu2structured(config, vtu_dir, stl_dir, mach)
                    example = create_tfExample(x, y, sim_config)
                    writer.write(example.SerializeToString())

                    j += 1
                except ValueError:
                    print('train, ValueError: ', vtu_dir)
                    continue
                except AttributeError:
                    print('train, Attribute error: ', vtu_dir)
                    continue


                #deletefiles(sample[0], sample[1])

        train_shards.append(j)

    for i in tqdm(range(n_files_test), desc='Test split', position=0):
        if test_remainder != 0 and i == n_files_test - 1:
            batch = [ds_test.pop(random.randrange(len(ds_test))) for _ in
                     range(test_remainder)]
        else:
            batch = [ds_test.pop(random.randrange(len(ds_test))) for _ in
                     range(config.preprocess.nsamples)]

        file_dir = os.path.join(config.preprocess.writedir,
                                'airfoilMNIST-test.tfrecord-{}-of-{}'.format(
                                    str(i).zfill(5),
                                    str(n_files_test).zfill(
                                        5)))

        with tf.io.TFRecordWriter(file_dir) as writer:
            j = 0
            for sample in tqdm(batch, desc='Shards', position=1,
                               leave=False):
                # split string to extract information about airfoil, angle of
                # attack and Mach number to write as feature into tfrecord
                sim_config = sample[0].rsplit('.', 1)[0]

                airfoil, angle, mach = sim_config.split('_')
                angle, mach = float(angle), float(mach)



                #copyfiles(sample[0], sample[1])
                vtu_dir = os.path.join(vtu_folder, sample[0])
                stl_dir = os.path.join(stl_folder, sample[1])

                try:
                    x, y = vtu2structured(config, vtu_dir, stl_dir, mach)
                    example = create_tfExample(x, y, sim_config)
                    writer.write(example.SerializeToString())

                    j += 1
                except ValueError:
                    print(vtu_dir)
                    continue
                except AttributeError:
                    print(vtu_dir)
                    continue
                #deletefiles(sample[0], sample[1])

        test_shards.append(j)

    # Create metadata files to read dataset with tfds.load(args)
    features = tfds.features.FeaturesDict({
        'encoder': tfds.features.Tensor(
            shape=(*config.vit.img_size, 1),
            dtype=np.float32,
        ),
        'decoder': tfds.features.Tensor(
            shape=(*config.vit.img_size, 9),
            dtype=np.float32,
        ),
        'label': tfds.features.Text(
            encoder=None,
            encoder_config=None,
            doc='Simulation config: airfoil_aoa_mach'
        ),
    })

    split_infos = [
        tfds.core.SplitInfo(
            name='train',
            shard_lengths=train_shards,
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name='test',
            shard_lengths=test_shards,
            num_bytes=0,
        ),
    ]

    tfds.folder_dataset.write_metadata(
        data_dir=config.preprocess.writedir,
        features=features,
        split_infos=split_infos,
        filename_template='{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}',
    )





