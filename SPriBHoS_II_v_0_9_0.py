# SPriBHoS-II Code (Spectral Primordial Black Hole Simulator-II)
# Developed by Albert Escrivà. https://github.com/albert-escriva
# This code is based on ArXiv:2504.05813
# VERSION v0.9.0 (preliminary release). Official v1.0.0 coming soon with much more functionalities.
# DISCLAIMER: THIS VERSION v0.9.0 IS FOR ILLUSTRATIVE PURPOSES 
# and is a simplified version of the one used in the paper.


# The current code corresponds to a simple example of a Gaussian-shaped profile \zeta
# with a peak amplitude μ = 2.0, which leads to type B PBH formation.
# The output of the code is a figure that illustratively shows the formation of trapping bifurcation horizons.

#to manage the number of cores from the PC (it will depend on your machine settings)
#os.environ['OPENBLAS_NUM_THREADS'] = '3'
#os.environ['MKL_NUM_THREADS'] = '3'
from numpy import *
import numpy as np
import math
from sympy import *
pi = math.pi
import time
import sys
np.seterr(divide='ignore', invalid='ignore')
from scipy.interpolate import CubicSpline
from numpy import linalg as LA
#External modulus
from Dmatrix import chebymatrix
#we import functions from the curvature file generator
from curvature_profiles_expq import curvature_profile_zeta
from curvature_profiles_expq import curvature_profile_zeta_der
from curvature_profiles_expq import curvature_profile_zeta_der2
from curvature_profiles_expq import curvature_profile_zeta_der_div_r
from curvature_profiles_expq import density_pert
from curvature_profiles_expq import velocity_pert
from curvature_profiles_expq import B_pert
from curvature_profiles_expq import K_initial
from matplotlib import pyplot as plt
start_time = time.time()
##############################

#Here we set up the initial variables and magnitudes.
w = 1./3. #EQ. of state. The case with the results of the paper corresponds to w=1/3
t_initial = 1.0 #initial time
alpha = 2./(3.*(1.+w))
#numerical initial conditions(background quantities)
H_bI = alpha/(t_initial) #Initial Hubble constant
e_bI = (3./(8.*pi))*H_bI**2 #initial energy density of the background
a_I = 1. #initial scale factor 
RHI = 1/H_bI # initial cosmological horizon
Nh = 100 #number of initial horizon radious, to set up the final point of the grid


r_initial=0.0 #initial point of the grid
r_final = Nh*RHI #final point of the grid, given by the mass enclosed for the given radious
rm_N = 10. #number of initial cosmological horizon that we put the length scale of the perturbtion rk. The long wavelength approximation must be fulfilld! Take rm_N always such that epsilon<0.1
rmww = rm_N*RHI #lenght-scale of the fluctuation
a1 = r_initial #initial point of the grid
a2 = 10.*rmww#final point of the grid

dt0 = 10**(-3.)#initial time-step
t_final = 10**6 #final time of the simulation
t = t_initial
#Differentiation matrix pseudospectral method

N1 = 500 #number of points for the Chebyshe grid. 
#In the current version of the code it is only implemented 1 single grid
D1,x1 = chebymatrix(N1,a1,a2) #Chebyshev differential matrix and set-up the grid
D1second = np.dot(D1,D1) #Chebyshev differential matrix for the second derivative

print ("Welcome to the primordial black hole simulator SPriBHoS-II")
print("The simulation corresponds to an example of type-B PBH for a Gaussian shape profile in zeta")
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
##background equation for energy density
def energy_FRW(t):
	e_FRWb = e_bI*(t_initial/t)**2
	return e_FRWb
#we apply the Newman boundary condition
def boundary_newman(uu,DD,index):
	uprime =0. #zero value of the derivative
	uii = uu[index]
	unew = uprime/DD[index,index] + uii - (1./DD[index,index])*np.dot(DD[index,:],uu)
	uu[index]=unew
	return uu

#system of partial differential equations we solve 
def system_rhs(Rpp1,Upp1,epp1,Bpp1,Gpp1,Kpp1,tt):
###########
###########
	efrw_new = energy_FRW(tt)
	Kpp1 = boundary_newman(Kpp1,D1,-1)
	epp1 = boundary_newman(epp1,D1,0)
	App1 = system_static(epp1,efrw_new)
	devUpp1,devRpp1,devBpp1,devApp1,devApp1_2 = compute_derivatives(D1,D1second,Upp1,Rpp1,Bpp1,App1)
	x1_term = devUpp1[-1]/devRpp1[-1]
	Gpp1[-1]=1.0
	Mpp1 = 0.5*Rpp1*(1.-Gpp1**2+Upp1**2)
	Mpp1[-1]=0.
######TIME-EVOLUTION EQUATIONS
	Utt1 = -App1*( 4.*pi*Rpp1*w*epp1 + Mpp1/(Rpp1**2)) +  devApp1*Gpp1/Bpp1 
	Rtt1 = Upp1*App1
	ett1 = epp1*(1.+w)*Kpp1*App1
	Gtt1 = devApp1*Upp1/Bpp1
	Btt1 = -App1*Bpp1*(Kpp1+2.*Upp1/Rpp1)
	Ktt1 = App1*(Kpp1+2.*Upp1/Rpp1)**2 + 2.0*App1*((Upp1/Rpp1)**2)+4.*pi*(1.+3.*w)*epp1*(App1) \
		-(1.0/(Bpp1**2))*(devApp1_2 + devApp1*(2.*devRpp1/Rpp1 - devBpp1/Bpp1))
#########
#########
	Utt1[-1]=0.
	Rtt1[-1]=0.
	Btt1[-1]=-App1[-1]*Bpp1[-1]*(Kpp1[-1]+2.*x1_term)
	Ktt1[-1]=0. #don't consider
	return Rtt1,Utt1,ett1,Btt1,Gtt1,Ktt1

#RK4 method
def rk4_step(Rpp1,Upp1,epp1,Bpp1,Gpp1,Kpp1,tt,dtt):
    k1_R1,k1_U1,k1_e1,k1_B1,k1_G1,k1_K1 =\
	system_rhs(Rpp1,Upp1,epp1,Bpp1,Gpp1,Kpp1,tt)
    k2_R1,k2_U1,k2_e1,k2_B1,k2_G1,k2_K1 = \
	system_rhs(Rpp1+0.5*dtt*k1_R1,Upp1+0.5*dtt*k1_U1,epp1+0.5*dtt*k1_e1,\
			Bpp1+0.5*dtt*k1_B1,Gpp1+0.5*dtt*k1_G1,Kpp1+0.5*dtt*k1_K1,tt+0.5*dtt)
    k3_R1,k3_U1,k3_e1,k3_B1,k3_G1,k3_K1 =\
	system_rhs(Rpp1+0.5*dtt*k2_R1,Upp1+0.5*dtt*k2_U1,epp1+0.5*dtt*k2_e1,\
			Bpp1+0.5*dtt*k2_B1,Gpp1+0.5*dtt*k2_G1,Kpp1+0.5*dtt*k2_K1,tt+0.5*dtt)
    k4_R1,k4_U1,k4_e1,k4_B1,k4_G1,k4_K1 =\
	system_rhs(Rpp1+dtt*k3_R1,Upp1+dtt*k3_U1,epp1+dtt*k3_e1,\
			Bpp1+dtt*k3_B1,Gpp1+dtt*k3_G1,Kpp1+dtt*k3_K1,tt+dtt)
    R_new1 = Rpp1 + (dtt / 6) * (k1_R1 + 2 * k2_R1 + 2 * k3_R1 + k4_R1)
    U_new1 = Upp1 + (dtt / 6) * (k1_U1 + 2 * k2_U1 + 2 * k3_U1 + k4_U1)
    e_new1 = epp1 + (dtt / 6) * (k1_e1 + 2 * k2_e1 + 2 * k3_e1 + k4_e1)
    B_new1 = Bpp1 + (dtt / 6) * (k1_B1 + 2 * k2_B1 + 2 * k3_B1 + k4_B1)
    G_new1 = Gpp1 + (dtt / 6) * (k1_G1 + 2 * k2_G1 + 2 * k3_G1 + k4_G1)
    K_new1 = Kpp1 + (dtt / 6) * (k1_K1 + 2 * k2_K1 + 2 * k3_K1 + k4_K1)
    return R_new1,U_new1,e_new1,B_new1,G_new1,K_new1

# we solve the lapse function
def system_static(epc,e_FRWc):
	Aqq = 1.*(e_FRWc/epc)**(w/(w+1.))
	return Aqq
#--------------------------------------------------------------------------------
# Initial perturbation magnitudes
def initial_perturbation_magnitudes(drhorho,dUU,dBB):
	dek = drhorho
	dUk = dUU
	dRk = -dek*w/((1+3.*w)*(1.+w))+dUk/(1+3.*w)
	dBk = dBB
	return dek,dUk,dRk,dBk
#We set up the initial conditions of the simulation
def initial_conditions(epskk,dekk,dRkk,dUkk,dBkk,zetak):
	e_Ikk = e_bI*(1.+dekk*(epskk**2))
	R_Ikk = a_I*x1*(np.exp(zetak))*(1.+dRkk*(epskk**2))              
	U_Ikk = H_bI*R_Ikk*(1.+dUkk*(epskk**2))
	B_Ikk =  a_I*(np.exp(zetak))*(1.+dBkk*(epskk**2)) 
	return e_Ikk,R_Ikk,U_Ikk,B_Ikk
#compaction function in the comoving gauge
def compact_function(Mv,Rv,efrw):
	Cc = 2*(Mv-(4./3.)*pi*efrw*Rv**3)/Rv
	Cc[-1]=0.
	return Cc
#The derivatives at each point are computed using Chebyshev differentiation matrix. 
def compute_derivatives(DDD,DDDsecond,Urr,Rrr,Brr,Arr):
	devU =  np.dot(DDD,Urr)
	devR =  np.dot(DDD,Rrr)
	devB = np.dot(DDD,Brr)
	devA = np.dot(DDD,Arr)
	devA_2 = np.dot(DDDsecond,Arr)
	return devU,devR,devB,devA,devA_2

def hamiltonian_constraint_initial1(DD,RR,ee,MM):
	constraint = (DD.dot(MM)-4.*pi*ee*(DD.dot(RR))*RR**2)
	return constraint

#main function: 
def search(mu_amplitude):
	epsrr = 1./(a_I*H_bI*rmww)
	#print ("amplitude de mu_aplitude que probamos ",mu_amplitude,"epse", epsrr )
	zeta_profile1 = curvature_profile_zeta(x1,mu_amplitude,rmww)
	#next three lines not used, you can check the initial condition for plotting
	zeta_profile1_der = curvature_profile_zeta_der(x1,mu_amplitude,rmww)
	zeta_profile1_der2 = curvature_profile_zeta_der2(x1,mu_amplitude,rmww)
	zeta_profile1_der_div = curvature_profile_zeta_der_div_r(x1,mu_amplitude,rmww)

	#initial perturbations
	drho1 = density_pert(x1,mu_amplitude,rmww,w)
	dU1 = velocity_pert(x1,mu_amplitude,rmww,w)
	dB1 = B_pert(x1,mu_amplitude,rmww,w) #this can be written in terms of rho and U as well (see the paper)
#####
	derr1,dUrr1,dRrr1,dBrr1 = initial_perturbation_magnitudes(drho1,dU1,dB1)
	e_Ie1,R_Ie1,U_Ie1,B_Ie1 = initial_conditions(epsrr,derr1,dRrr1,dUrr1,dBrr1,zeta_profile1)

#fixing the other initial conditions
	G_Ie1 = np.dot(D1,R_Ie1)/B_Ie1
	M_Ie1 = 0.5*R_Ie1*(1.-G_Ie1**2+U_Ie1**2)

###we can put the initial condition for K from the quasi-homogeneus solution, 
	#Ktilde1 = -drho1/(3.*(1.+w))
	#KK_Ie1 = -3.*H_bI*(1.+(epsrr**2)*Ktilde1)
###but we use the following equation, see the paper.
	KK_Ie1 = K_initial(R_Ie1,U_Ie1,e_Ie1,B_Ie1,M_Ie1,np.dot(D1,G_Ie1))

	Rvv1,evv1,Uvv1,Gvv1,Bvv1,Kvv1 = R_Ie1,e_Ie1,U_Ie1,G_Ie1,B_Ie1,KK_Ie1

	#set initial
	t = t_initial

	#start
	while t<t_final: 

		dt = dt0*t**(alpha) #conformal time-step
		thresholdn = dt/10

		Rnn1,Unn1,enn1,Bnn1,Gnn1,Knn1 = rk4_step(Rvv1,Uvv1,evv1,Bvv1,Gvv1,Kvv1,t,dt)
		if np.isnan(enn1[-5]) == True:
			print ("Divergence found, stop",t)

		t += dt
		e_FRW_time = energy_FRW(t)

		if (t%store_step<thresholdn) or (t%store_step>(store_step-thresholdn)):

			# Misner-Sharp mass
			Mnn1 = 0.5*Rnn1*(1.-Gnn1**2+Unn1**2)

			#compactness
			compactness_1 = 2.*Mnn1/Rnn1
			compactness_1[-1]=0.

			spline_compactness = CubicSpline(x1[::-1],compactness_1[::-1]-1)
			#we find the points where 2M=R (horizons)
			root_compactness = spline_compactness.roots(discontinuity=False , extrapolate=False)

			for l in range(len(root_compactness)):
				fh.write( str(t)+" "+str(root_compactness[l])+" "   )
				fh.write("\n")

			Compact = compact_function(Mnn1,Rnn1,e_FRW_time) #we construct the compaction function
			Cmax = np.amax(Compact)
			if Cmax>2.0:
				print("We have stopped the simulation")
				fh.close()
				break
################
################
	##we replace the new variables
		Rvv1,evv1,Uvv1,Gvv1,Bvv1,Kvv1 = Rnn1[:],enn1[:],Unn1[:],Gnn1[:],Bnn1[:],Knn1[:]
#--------------------------------------------------------------------------------
mu_value = 2.0 #value of the peak of the curvature \zeta
store_step = 0.01 #storing criteria of valus in the loop while

#file to store the horizons 2M=R
filename_horizons = "horizons.dat" 
fh = open(filename_horizons, "w")


start_time = time.time()

#we start the method for numerically evolving the collapse of the fluctuations
search(mu_value)


print ("Simulation done successfully. The time of the computation was:")
print("--- %s seconds ---"% (time.time()-start_time))

##now let's plot the result of the simulation, for what we observe a type B PBH
data_horizons = np.loadtxt(filename_horizons) 
time_h = data_horizons[:,0] 
r_h = data_horizons[:,1] 
#simple plot
plt.xlabel(r'$r_{*}$')
plt.ylabel(r'$t$')
plt.plot(r_h , time_h , linestyle=' ' , marker='o',markersize=1)
plt.grid()
plt.show()


