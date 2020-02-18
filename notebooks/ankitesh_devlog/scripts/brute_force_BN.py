

import sys
sys.path.insert(1,"/home/ankitesh/miniconda3/envs/CbrainCustomLayer/lib/python3.6/site-packages") #work around for h5py
from cbrain.imports import *
from cbrain.cam_constants import *
from cbrain.utils import *
from cbrain.layers import *
from cbrain.data_generator import DataGenerator
import tensorflow as tf
from tensorflow import math as tfm
# import tensorflow_probability as tfp
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import xarray as xr
import numpy as np
from cbrain.model_diagnostics import ModelDiagnostics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as imag
import scipy.integrate as sin
# import cartopy.crs as ccrs
import matplotlib.ticker as mticker
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle

TRAINDIR = '/oasis/scratch/comet/ankitesh/temp_project/PrepData/'
TRAINFILE = 'CI_SP_M4K_train_shuffle.nc'
NORMFILE = 'CI_SP_M4K_NORM_norm.nc'
VALIDFILE = 'CI_SP_M4K_valid.nc'

scale_dict = load_pickle('/home/ankitesh/CBrain_project/CBRAIN-CAM/nn_config/scale_dicts/009_Wm2_scaling.pkl')
in_vars = ['QBP','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']


train_gen = DataGenerator(
    data_fn = TRAINDIR+TRAINFILE,
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+NORMFILE,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True,
    normalize_flag=False
)

valid_gen = DataGenerator(
    data_fn = TRAINDIR+VALIDFILE,
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+NORMFILE,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True,
    normalize_flag=False
)



inp = Input(shape=(64,))
batch_norm_1 = BatchNormalization()(inp)
densout = Dense(128, activation='linear')(batch_norm_1)
densout = LeakyReLU(alpha=0.3)(densout)
for i in range (6):
    batch_norm_i = BatchNormalization()(densout)
    densout = Dense(128, activation='linear')(batch_norm_i)
    densout = LeakyReLU(alpha=0.3)(densout)
batch_norm_out = BatchNormalization()(densout)
out = Dense(64, activation='linear')(batch_norm_out)
Brute_force = tf.keras.models.Model(inp, out)


Brute_force.summary()


path_HDF5 = '/oasis/scratch/comet/ankitesh/temp_project/'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save_BF_BN = ModelCheckpoint(path_HDF5+'CI01_BF_BN.hdf5',save_best_only=True, monitor='val_loss', mode='min')


Brute_force.compile(tf.keras.optimizers.Adam(), loss=mse)

Nep = 10
Brute_force.fit_generator(train_gen, epochs=Nep, validation_data=valid_gen,\
              callbacks=[earlyStopping, mcp_save_BF_BN])
