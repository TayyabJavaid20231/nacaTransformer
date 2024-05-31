import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.dataset =  '/local/disk1/tjavaid/PreprocessedDataSet4nacaFOAM/Incompressible_128x128/'
    #config.dataset =  '/work/tjavaid/tjavaid/directInterpolation/src4Fixed_k_Interpolation/sample4Comparison256/'
    config.output_dir = '/local/disk1/tjavaid/Plots/plots4AllStandardizerandom'
    config.trainer = 'train_parallel'  # 'train' or train_parallel or 'preprocess' or 'inference' or 'postprocess_state'
    config.num_epochs = 50
    config.batch_size = 24
    config.shuffle_buffer_size = 1024
    config.learning_rate_scheduler = "sgdr"
    config.learning_rate_end_value = 1e-5
    config.sgdr_restarts = 1 #int(config.num_epochs / 50)
    config.warmup_fraction = 0.1
    config.weight_decay = 0.1
    config.output_frequency = 10

    config.vit = ml_collections.ConfigDict()
    config.vit.img_size = (256, 256)
    config.vit.patch_size = (4, 4)  # num_patches = (img_size / patch_size)
    config.vit.hidden_size = 450  # patch_size^2 * num_channels
    config.vit.num_layers = 3
    config.vit.num_heads = 3
    config.vit.dim_mlp = 4 * config.vit.hidden_size
    config.vit.dropout_rate = 0.0
    config.vit.att_dropout_rate = 0.0

    config.preprocess = ml_collections.ConfigDict()
    config.preprocess.readdir = ''
    config.preprocess.writedir = ''
    config.preprocess.incompressible = True
    config.preprocess.train_split = 0.8
    config.preprocess.aoa = (-5, 15)
    config.preprocess.mach = (0, 0.3)
    config.preprocess.nsamples = 10
    config.preprocess.dim = (-0.75, 1.25, -1, 1)
    config.preprocess.num_neighbors = 5
    config.preprocess.gpu_id = 0

    config.pressure_preprocessing = ml_collections.ConfigDict()
    config.pressure_preprocessing.enable = True
    config.pressure_preprocessing.type = 'standardize_all'   # standardize or standardize_all or range or coefficient or 2_standard_deviation_normalization
    config.pressure_preprocessing.new_range = (0, 1)
    
    config.internal_geometry = ml_collections.ConfigDict()
    config.internal_geometry.set_internal_value = False
    config.internal_geometry.value = 0
    config.internal_geometry.set_postProcessed_internal_value = True
    config.internal_geometry.postProcessed_value = 0

    config.plots = ml_collections.ConfigDict()
    config.plots.set_range = False
    config.plots.pressure_range = (0.975, 1.015)
    config.plots.ux_range = (0.5, 1.2)
    config.plots.uy_range = (-0.25, 0.25)

    config.denormalization = ml_collections.ConfigDict()
    config.denormalization.denormalize = True

    return config
