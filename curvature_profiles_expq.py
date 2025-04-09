
import numpy as np
import math
import matplotlib.pyplot as plt  
pi = math.pi
##############
### In this file, we set up the initial conditions.
### The initial condition corresponds to a 
### Gaussian-shaped profile (see the paper).


##zeta profile
def curvature_profile_zeta(rf,muf,rmf):
        xf = rf/rmf;
        zeta_g = muf*np.exp(-(xf)**(2.));
        return zeta_g
##derivative of zeta
def curvature_profile_zeta_der(rf,muf,rmf):
        xf = rf/rmf;
        output=(-2.*((np.exp(((-xf**(2.)))))*(((xf**(-1.+(2.)))*muf))))/rmf;
        return output
#second derivative of zeta
def curvature_profile_zeta_der2(rf,muf,rmf):
        xf = rf/rmf
        output=2.*((np.exp(((-(xf**2)))))*((rmf**-2.)*((-1.+(2.*(xf**2)))*muf)));        
        return output
####ratio zeta'/r
def curvature_profile_zeta_der_div_r(rf,muf,rmf):
        xf = rf/rmf
        output=-2.*((np.exp(((-(xf**2)))))*((rmf**-2.)*muf));              
        return output
########## initial conditions for hydrodynamical variables
def density_pert(xf,muf,rmf,w):
        zeta_profile = curvature_profile_zeta(xf,muf,rmf)
        zeta_der = curvature_profile_zeta_der(xf,muf,rmf)
        zeta_der2 = curvature_profile_zeta_der2(xf,muf,rmf)
        zeta_der_div_r = curvature_profile_zeta_der_div_r(xf,muf,rmf)
        factor_zetas = zeta_der2 + 2.*zeta_der_div_r + 0.5*zeta_der**2
        output = ((-2.*(1.+w))/(5.+(3.*w)))*np.exp(-2*zeta_profile)*factor_zetas*(rmf**2);
        return output
def velocity_pert(xf,muf,rmf,w):
        zeta_profile = curvature_profile_zeta(xf,muf,rmf)
        zeta_der = curvature_profile_zeta_der(xf,muf,rmf)
        zeta_der_div_r = curvature_profile_zeta_der_div_r(xf,muf,rmf)
        factor_zetas = 2.*zeta_der_div_r + zeta_der**2
        output = (1./(5.+(3.*w)))*np.exp(-2*zeta_profile)*factor_zetas*(rmf**2);
        return output
def B_pert(xf,muf,rmf,w):
        zeta_profile = curvature_profile_zeta(xf,muf,rmf)
        zeta_der = curvature_profile_zeta_der(xf,muf,rmf)
        zeta_der2 = curvature_profile_zeta_der2(xf,muf,rmf)
        zeta_der_div_r = curvature_profile_zeta_der_div_r(xf,muf,rmf)
        factor_zetas = 4.*w*zeta_der_div_r + (w-1)*(zeta_der**2) + 2.*(1.+w)*zeta_der2
        output = (((5.+(3.*w))**-1.)/(1.+(3.*w)))*np.exp(-2*zeta_profile)*factor_zetas*(rmf**2);
        return output
#initial condition for K we have used (see the paper)
def K_initial(RR,UU,ee,BB,MM,derGG):        
        output = ( (-1./UU)*(4.*pi*ee*RR - (MM/RR**2) + derGG/(BB) ) - 2.*UU/RR ) 
        output[-1]=0. #we will apply NBC
        return output

