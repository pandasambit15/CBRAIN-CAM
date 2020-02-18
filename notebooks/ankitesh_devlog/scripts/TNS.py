import sys
sys.path.insert(1,"/home1/07064/tg863631/anaconda3/envs/CbrainCustomLayer/lib/python3.6/site-packages") #work around for h5py
from cbrain.imports import *
from cbrain.cam_constants import *
from cbrain.utils import *
from cbrain.layers import *
from cbrain.data_generator import DataGenerator
import tensorflow as tf
import tensorflow.math as tfm
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




class T2TmTNS(Layer):
    def __init__(self, inp_subT, inp_divT, inp_subTNS, inp_divTNS, hyam, hybm, **kwargs):
        """
        From temperature to (temperature)-(near-surface temperature)
        Call using ([input])
        Assumes
        prior: [QBP, 
        TBP, 
        PS, SOLIN, SHFLX, LHFLX]
        Returns
        post(erior): [QBP,
        TfromNS, 
        PS, SOLIN, SHFLX, LHFLX]
        Arguments:
        inp_subT = Normalization based on input with temperature (subtraction constant)
        inp_divT = Normalization based on input with temperature (division constant)
        inp_subTNS = Normalization based on input with (temp - near-sur temp) (subtraction constant)
        inp_divTNS = Normalization based on input with (temp - near-sur temp) (division constant)
        hyam = Constant a for pressure based on mid-levels
        hybm = Constant b for pressure based on mid-levels
        """
        self.inp_subT, self.inp_divT, self.inp_subTNS, self.inp_divTNS, self.hyam, self.hybm = \
            np.array(inp_subT), np.array(inp_divT), np.array(inp_subTNS), np.array(inp_divTNS), \
        np.array(hyam), np.array(hybm)
        # Define variable indices here
        # Input
        self.QBP_idx = slice(0,30)
        self.TBP_idx = slice(30,60)
        self.PS_idx = 60
        self.SHFLX_idx = 62
        self.LHFLX_idx = 63

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = {'inp_subT': list(self.inp_subT), 'inp_divT': list(self.inp_divT),
                  'inp_subTNS': list(self.inp_subTNS), 'inp_divTNS': list(self.inp_divTNS),
                  'hyam': list(self.hyam),'hybm': list(self.hybm)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, arrs):
        prior = arrs
        
        Tprior = prior[:,self.TBP_idx]*self.inp_divT[self.TBP_idx]+self.inp_subT[self.TBP_idx]
        
        Tile_dim = tf.constant([1,30],tf.int32)
        TNSprior = ((Tprior-tf.tile(tf.expand_dims(Tprior[:,-1],axis=1),Tile_dim))-\
                    self.inp_subTNS[self.TBP_idx])/\
        self.inp_divTNS[self.TBP_idx]
        
        post = tf.concat([prior[:,:30],tf.cast(TNSprior,tf.float32),prior[:,60:]], axis=1)
        
        return post

    def compute_output_shape(self,input_shape):
        """Input shape + 1"""
        return (input_shape[0][0])




scale_dict = load_pickle('/home1/07064/tg863631/CBrain_project/CBRAIN-CAM/nn_config/scale_dicts/009_Wm2_scaling.pkl')

in_vars = ['QBP','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']


TRAINDIR = '/scratch/07064/tg863631/PrepData/'
TRAINFILE = 'CI_SP_M4K_train_shuffle.nc'
NORMFILE = 'CI_SP_M4K_NORM_norm.nc'
VALIDFILE = 'CI_SP_M4K_valid.nc'


train_gen = DataGenerator(
    data_fn = TRAINDIR+TRAINFILE,
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+NORMFILE,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True
)

valid_gen = DataGenerator(
    data_fn = TRAINDIR+VALIDFILE,
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+NORMFILE,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True
)
in_vars = ['QBP','TfromNS','PS', 'SOLIN', 'SHFLX', 'LHFLX']
out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']
TRAINFILE_TNS = 'CI_TNS_M4K_NORM_train_shuffle.nc'
NORMFILE_TNS = 'CI_TNS_M4K_NORM_norm.nc'
VALIDFILE_TNS = 'CI_TNS_M4K_NORM_valid.nc'
train_gen_TNS = DataGenerator(
    data_fn = TRAINDIR+TRAINFILE_TNS,
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+NORMFILE_TNS,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=1024,
    shuffle=True
)


path = '/home1/07064/tg863631/CBrain_project/CBRAIN-CAM/cbrain/'
path_hyam = 'hyam_hybm.pkl'
hf = open(path+path_hyam,'rb')
hyam,hybm = pickle.load(hf)

print(train_gen_TNS[50][0].shape)
print(train_gen_TNS[50][1].shape)
print(train_gen_TNS[78][0].shape)
print(train_gen_TNS[78][1].shape)


inp = Input(shape=(64,))
inpTNS = T2TmTNS(inp_subT=train_gen.input_transform.sub, 
              inp_divT=train_gen.input_transform.div, 
              inp_subTNS=train_gen_TNS.input_transform.sub, 
              inp_divTNS=train_gen_TNS.input_transform.div, 
              hyam=hyam, hybm=hybm)(inp)
densout = Dense(128, activation='linear')(inpTNS)
densout = LeakyReLU(alpha=0.3)(densout)
for i in range (6):
    densout = Dense(128, activation='linear')(densout)
    densout = LeakyReLU(alpha=0.3)(densout)
out = Dense(64, activation='linear')(densout)
Input_TNS = tf.keras.models.Model(inp, out)


Input_TNS.summary()

path_HDF5 = '/scratch/07064/tg863631/models/'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save_TNS = ModelCheckpoint(path_HDF5+'CI01_TNS.hdf5',save_best_only=True, monitor='val_loss', mode='min')


Input_TNS.compile(tf.keras.optimizers.Adam(), loss=mse)

Nep = 2
Input_TNS.load_weights(path_HDF5+'CI01_TNS.hdf5')
Input_TNS.fit_generator(train_gen, epochs=Nep, validation_data=valid_gen,\
              callbacks=[earlyStopping, mcp_save_TNS])
