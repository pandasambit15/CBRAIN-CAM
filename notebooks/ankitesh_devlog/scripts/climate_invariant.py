
from cbrain.imports import *
from tensorflow.keras.layers import *

def eliq(T):
    a_liq = np.float32(np.array([-0.976195544e-15,-0.952447341e-13,\
                                 0.640689451e-10,\
                      0.206739458e-7,0.302950461e-5,0.264847430e-3,\
                      0.142986287e-1,0.443987641,6.11239921]));
    c_liq = np.float32(-80.0)
    T0 = np.float32(273.16)
    return np.float32(100.0)*tfm.polyval(a_liq,tfm.maximum(c_liq,T-T0))

def eice(T):
    a_ice = np.float32(np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,\
                      0.602588177e-7,0.615021634e-5,0.420895665e-3,\
                      0.188439774e-1,0.503160820,6.11147274]));
    c_ice = np.float32(np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07]))
    T0 = np.float32(273.16)
    return tf.where(T>c_ice[0],eliq(T),\
                   tf.where(T<=c_ice[1],np.float32(100.0)*(c_ice[3]+tfm.maximum(c_ice[2],T-T0)*\
                   (c_ice[4]+tfm.maximum(c_ice[2],T-T0)*c_ice[5])),\
                           np.float32(100.0)*tfm.polyval(a_ice,T-T0)))

def esat(T):
    T0 = np.float32(273.16)
    T00 = np.float32(253.16)
    omtmp = (T-T00)/(T0-T00)
    omega = tfm.maximum(np.float32(0.0),tfm.minimum(np.float32(1.0),omtmp))

    return tf.where(T>T0,eliq(T),tf.where(T<T00,eice(T),(omega*eliq(T)+(1-omega)*eice(T))))

def qv(T,RH,P0,PS,hyam,hybm):
    
    R = np.float32(287.0)
    Rv = np.float32(461.0)
    p = P0 * hyam + PS[:, None] * hybm # Total pressure (Pa)
    
    T = tf.cast(T,tf.float32)
    RH = tf.cast(RH,tf.float32)
    p = tf.cast(p,tf.float32)
    
    return R*esat(T)*RH/(Rv*p)
    # DEBUG 1
    # return esat(T)
    
def RH(T,qv,P0,PS,hyam,hybm):
    R = np.float32(287.0)
    Rv = np.float32(461.0)
    p = P0 * hyam + PS[:, None] * hybm # Total pressure (Pa)
    
    T = tf.cast(T,tf.float32)
    qv = tf.cast(qv,tf.float32)
    p = tf.cast(p,tf.float32)
    
    return Rv*p*qv/(R*esat(T))

class QV2RH(Layer):
    def __init__(self, inp_subQ, inp_divQ, inp_subRH, inp_divRH, hyam, hybm, **kwargs):
        """
        Call using ([input])
        Assumes
        prior: [QBP, 
        TBP, PS, SOLIN, SHFLX, LHFLX]
        Returns
        post(erior): [RHBP,
        TBP, PS, SOLIN, SHFLX, LHFLX]
        Arguments:
        inp_subQ = Normalization based on input with specific humidity (subtraction constant)
        inp_divQ = Normalization based on input with specific humidity (division constant)
        inp_subRH = Normalization based on input with relative humidity (subtraction constant)
        inp_divRH = Normalization based on input with relative humidity (division constant)
        hyam = Constant a for pressure based on mid-levels
        hybm = Constant b for pressure based on mid-levels
        """
        self.inp_subQ, self.inp_divQ, self.inp_subRH, self.inp_divRH, self.hyam, self.hybm = \
            np.array(inp_subQ), np.array(inp_divQ), np.array(inp_subRH), np.array(inp_divRH), \
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
        config = {'inp_subQ': list(self.inp_subQ), 'inp_divQ': list(self.inp_divQ),
                  'inp_subRH': list(self.inp_subRH), 'inp_divRH': list(self.inp_divRH),
                  'hyam': list(self.hyam),'hybm': list(self.hybm)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, arrs):
        prior = arrs
        
        Tprior = prior[:,self.TBP_idx]*self.inp_divQ[self.TBP_idx]+self.inp_subQ[self.TBP_idx]
        qvprior = prior[:,self.QBP_idx]*self.inp_divQ[self.QBP_idx]+self.inp_subQ[self.QBP_idx]
        PSprior = prior[:,self.PS_idx]*self.inp_divQ[self.PS_idx]+self.inp_subQ[self.PS_idx]
        RHprior = (RH(Tprior,qvprior,P0,PSprior,self.hyam,self.hybm)-\
                    self.inp_subRH[self.QBP_idx])/self.inp_divRH[self.QBP_idx]
        
        post = tf.concat([tf.cast(RHprior,tf.float32),prior[:,30:]], axis=1)
        
        return post

    def compute_output_shape(self,input_shape):
        """Input shape + 1"""
        return (input_shape[0][0])


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
