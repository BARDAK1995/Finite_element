from pylab import *
import numpy.linalg 
import numpy as np
import matplotlib.pyplot as plt

Ro=2
Rn=1
K1=1
K2=4
g0=1
To=0

def AnalyticSpherical(Ro,Rn,K1,K2,g0,To,nElements=1000):
	def TD1(T1,a,r,Rn):return T1+a*(1-(r/Rn)**2)
	def TD2(T0,b,R0,r):return T0+b*((R0/r)-1)
	dom1=arange(0,Rn,Rn/nElements);dom2=arange(Rn,Ro,(Ro-Rn)/nElements)
	rlist=hstack([dom1,dom2])
	Discontinuity_index=len(dom1)

	q1_flux=Rn*g0/3
	b=q1_flux*(Rn**2)/(K2*Ro); a=g0*(Rn**2)/(6*K1)
	T1=To+b*((Ro/Rn)-1)
	td1=TD1(T1,a,rlist[:Discontinuity_index],Rn)
	td2=TD2(To,b,Ro,rlist[Discontinuity_index:])
	Tlist=np.hstack([td1,td2])
	return rlist,Tlist
	
rlist,Tlist=AnalyticSpherical(Ro,Rn,K1,K2,g0,To)
figure();plot(rlist,Tlist);show()

