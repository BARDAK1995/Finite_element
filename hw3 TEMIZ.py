from pylab import *
import numpy.linalg 
import numpy as np
import matplotlib.pyplot as plt

def AnalyticSpherical(Ro,Rn,K1,K2,g0,To,nElements=1000):
	def TD1(T1,a,r,Rn):return T1+a*(1-(r/Rn)**2)
	def TD2(T0,b,R0,r):return T0+b*((R0/r)-1)
	dom1=arange(0,Rn,Rn/nElements);dom2=arange(Rn,Ro,(Ro-Rn)/nElements)
	rlist=hstack([dom1,dom2]);Discontinuity_index=len(dom1)
	q1_flux=Rn*g0/3
	b=q1_flux*(Rn**2)/(K2*Ro); a=g0*(Rn**2)/(6*K1)
	T1=To+b*((Ro/Rn)-1)
	td1=TD1(T1,a,rlist[:Discontinuity_index],Rn)
	td2=TD2(To,b,Ro,rlist[Discontinuity_index:])
	Tlist=np.hstack([td1,td2])
	return rlist,Tlist
	
def AnalyticSphericalError(Ro,Rn,K1,K2,g0,To,dom1,dom2):
	def TD1(T1,a,r,Rn):return T1+a*(1-(r/Rn)**2)
	def TD2(T0,b,R0,r):return T0+b*((R0/r)-1)
	rlist=hstack([dom1,dom2]);Discontinuity_index=len(dom1)
	q1_flux=Rn*g0/3
	b=q1_flux*(Rn**2)/(K2*Ro); a=g0*(Rn**2)/(6*K1)
	T1=To+b*((Ro/Rn)-1)
	td1=TD1(T1,a,rlist[:Discontinuity_index],Rn)
	td2=TD2(To,b,Ro,rlist[Discontinuity_index:])
	Tlist=np.hstack([td1,td2])
	return Tlist
	
def getError(r_case,r_core,k_core,k_clad,g0,T_o,Mesh,dsc_index,ux_g_t,verbose=True):
	Mesh1=Mesh[:dsc_index];Mesh2=Mesh[dsc_index:]
	nelem=len(Mesh)
	errors=(abs(AnalyticSphericalError(r_case,r_core,k_core,k_clad,g0,T_o,Mesh1,Mesh2)-ux_g_t))/(len(Mesh)+len(Mesh1))
	dummy,U_exact=AnalyticSpherical(r_case,r_core,k_core,k_clad,g0,T_o)
	if verbose:
		figure();plot(Mesh,ux_g_t,linewidth=2,linestyle='--',color="red");plot(dummy,U_exact,linewidth=1,color="black");
		title(f'EXACT vs F.E.')
		figtext(0.2,0.7,f'TOTAL ERROR IS  {sum(errors)}')
		ylabel(f'TEMP')
		legend([f'F.E with {nelem} elements','EXACT solution'])
		show()
	return sum(errors), errors
	
def solveThomas1(f,e,g,r):
	'''
	[f1  g1 0  0    ]
	[e1  f2 g2 0    ]
	[0   e2 f3 g3    ]
	r=results=b
	'''
	ndim=len(r)
	for i in range(ndim-1):
		e[i]/=f[i]
		f[i+1]-=e[i]*g[i]
		r[i+1]-=e[i]*r[i]
	x=r/f
	rf=r/f
	gf=g/f[:-1]
	for i2 in range(ndim-2,-1,-1):
		x[i2]=rf[i2]-gf[i2]*x[i2+1]
	return x

def MesherConstant(xL,xR,nElements):
	elemLen=(xR-xL)/nElements
	Nodes=arange(xL,xR+elemLen/2,elemLen)
	return Nodes
	
def MesherConstantDiscontinous(xL,xMid,xR,nElements,vars=[]):
	n_elem1=int((nElements)/2); n_elem2=nElements-n_elem1
	range1=xMid-xL; range2=xR-xMid
	elemLen1=(xMid-xL)/n_elem1; elemLen2=(xR-xMid)/n_elem2
	Nodes1=arange(xL,xMid,elemLen1)
	Nodes2=arange(xMid,xR+elemLen2/2,elemLen2)
	Nodes=append(Nodes1,Nodes2)
	discVars=0
	if isinstance(vars, np.ndarray):
		DiscVars1=ones_like(Nodes1)*vars[:,:,0]
		DiscVars2=ones_like(Nodes2)*vars[:,:,1]
		discVars=hstack([DiscVars1,DiscVars2[:,:-1]])
	return Nodes,discVars,n_elem1

def build_elemental_coefficient_TENSOR_spherical_ANALYTIC(k_list,rmean_list,element_lengths,tridiag=False):
	Hv1=(k_list/element_lengths) * ( (rmean_list**2) + (element_lengths**2) /12); 
	#Hv2=element_lengths;
	T1=array([[1,-1],[-1,1]])
	#T2=array([[c/3,c/6],[c/6,c/3]])
	if not tridiag: TensorT=np.array([T1]);Tensor_vec=np.array([Hv1])
	else:  TensorT=T1.flatten();Tensor_vec=Hv1
	#builds element Coefficient matrices
	if tridiag: K_elements_TENSOR=einsum('j,i->ij',TensorT,Tensor_vec)
	else:       K_elements_TENSOR=tensordot(transpose(Tensor_vec),(TensorT),1)
	return K_elements_TENSOR
	
def elementalVariableFunction_Rsquare(local_pshyi,meanValues=0,lengths=0):
	return (meanValues+(lengths/2)*local_pshyi)**2

def build_elemental_coefficient_TENSOR_GAUSS(Elemental_constants,rmean_list,element_lengths,Element_Variable_func=elementalVariableFunction_Rsquare,Aprox_Func_Matrix=array([[1/4,-1/4],[-1/4,1/4]]),n=2,tridiag=False):
	if n==1:w=[2];phyy=[0]
	if n==2:w=[1,1];phyy=[-1/(3**0.5),1/(3**0.5)]
	if n==3:w=[5/9,8/9,5/9];phyy=[-((3/5)**0.5),0,(3/5)**0.5]
	if tridiag: Aprox_Func_Matrix=Aprox_Func_Matrix.flatten()
	Rs=zeros([n,len(element_lengths)])
	Elem_CONSTANT=tensordot(Elemental_constants,Aprox_Func_Matrix,0)
	for ns in range(n):
		Rs[ns]=(Element_Variable_func(phyy[ns],rmean_list,element_lengths))*w[ns]
	Rsum=sum(Rs,0)
	if tridiag: K_elements_TENSOR=einsum('ij,i->ij',Elem_CONSTANT,Rsum)
	else:K_elements_TENSOR=einsum('ijk,i->ijk',Elem_CONSTANT,Rsum)
	return K_elements_TENSOR
	
def build_GLOBAL_Matrices(K_tensor,f_elemental,tridiag=False):
	'''hocam loop kullanmayi kendime gecen yil yasakladigim icin bu fonksiyon var
	direk local elementler K tensorunu Global K matrixine mapliyorum
	'''
	if not tridiag:
		nElements=K_tensor[:,0,0].size
		T_local2Global=generateTransformationTensors(nElements)
		TransT=transpose(T_local2Global)
		rhs=einsum('ijk,kai->ija',K_tensor,T_local2Global)
		K_global=einsum('ijk,ikb->jb',TransT,rhs)
	else:
		nElements=K_tensor.shape[0]
		f=np.zeros(nElements+1);f[0]=K_tensor[0,0];f[-1]=K_tensor[-1,3]
		c=K_tensor[:,2]
		b=K_tensor[:,1]
		f[1:-1]=K_tensor[1:,0]+K_tensor[:-1,3]
		K_global=array([c,f,b])
	F_global=transpose([append(0,f_elemental[1,:,:])+append(f_elemental[0,:,:],0)])
	return K_global,F_global

def generateTransformationTensors(nElements):
	Tensor_Local_2_Global=zeros((2,nElements+1,nElements))
	i2=identity(2)
	elems=[[i,i+2] for i in range(nElements)]
	for i in elems:
		Tensor_Local_2_Global[:,i[0]:i[1],i[0]]=i2
	return Tensor_Local_2_Global

def RHS_elemental_forHW1(nodes,ElemLengths):
	x_a=nodes[:-1]
	x_b=nodes[1:]
	nElements=ElemLengths.size
	fM1=array([[-sin(4*x_a)],[sin(4*x_b)]])
	DummyVector=(cos(4*x_b)-cos(4*x_a))/(4*ElemLengths)
	fM2=array([[-1*DummyVector],[1*DummyVector]])
	return (3/4)*(fM1+fM2)
	
def RHS_elemental_forHW3_analytic(ElemLengths,Rmean_elems,qlist):
	nElements=ElemLengths.size
	c1=(ElemLengths*qlist/6)
	c2=3*(Rmean_elems**2)+(ElemLengths**2)/4
	c3=Rmean_elems*ElemLengths
	c12=c1*c2; c13=c1*c3
	fM1=array([[c12],[c12]]); fM2=array([[-c13],[c13]])
	return (fM1+fM2)
	
def RHS_elemental_forHW3_Gauss(Elemental_RHS_constants,rmean_list,element_lengths,Aprox_functions,Element_Variable_func=elementalVariableFunction_Rsquare,n=2):
	if n==1:w=array([2]);phyy=array([0])
	if n==2:w=array([1,1]);phyy=array([-1/(3**0.5),1/(3**0.5)])
	if n==3:w=array([5/9,8/9,5/9]);phyy=array([-((3/5)**0.5),0,(3/5)**0.5])
	Rs=zeros([n,len(element_lengths)])
	AproxF_phy_Matrix=Aprox_functions(phyy)
	Elemental_RHS_tensor=tensordot(Elemental_RHS_constants,AproxF_phy_Matrix,0)
	for ns in range(n):
		Rs[ns]=(Element_Variable_func(phyy[ns],rmean_list,element_lengths))*w[ns]
	rhs1=einsum('ijk,aki->jai',Elemental_RHS_tensor,array([Rs]))
	return rhs1
	
def build_naturalBC_term(nElem,naturalBC,verbose=False):
	Q=np.zeros(nElem+1)
	SecondaryVariableLocations=[0,1]
	for item in naturalBC:
		Q[item[0]*nElem]=item[1]
		if item[0] in SecondaryVariableLocations:
			SecondaryVariableLocations.remove(item[0])
	SecondaryVariableLocations=array(SecondaryVariableLocations)*nElem
	if verbose:print(f'Q BC vector is\n {Q.reshape(nElem+1,1)}\n secondary variable Locations are \n {SecondaryVariableLocations}\n')
	return Q.reshape(nElem+1,1), SecondaryVariableLocations

def SolverPrimary(K_primary,RHS_primary,Essential_BC,tridiag=False):
	if tridiag: return SolverPrimary_TriDiagonal(K_primary,RHS_primary,Essential_BC)
	else:       return SolverPrimary_FULL_MATRIX(K_primary,RHS_primary,Essential_BC)

def SolverPrimary_FULL_MATRIX(K_primary,RHS_primary,Essential_BC):
	RHS_primary=RHS_primary.flatten()
	U=solve(K_primary,RHS_primary)
	for node in Essential_BC:
		nodeIndex=node[0]
		if node[0]==0: U=append(node[1],U)
		else: U=append(U,node[1])
	return U
	
def SolverPrimary_TriDiagonal(K_primary,RHS_primary,Essential_BC):
	RHS_primary=RHS_primary.flatten()
	e,f,g=K_primary
	U=solveThomas1(f,e,g,RHS_primary)
	for node in Essential_BC:
		nodeIndex=node[0]
		if node[0]==0: U=append(node[1],U)
		else: U=append(U,node[1])
	return U

def Solverinitialize(K_global,RHS,essentialBC,sec_loc,tridiag=False):
	if tridiag:return Solverinitialize_Thomas(K_global,RHS,essentialBC,sec_loc)
	else:      return SolverinitializeFULLmatrix(K_global,RHS,essentialBC,sec_loc)
	
def Solverinitialize_Thomas(K_global,RHS,essentialBC,sec_loc):
	c,f,b=K_global.copy()
	RHS_primary=RHS.copy()
	nElements=c.size
	dummyVector=np.zeros(nElements+1)
	for knownNodevalue in essentialBC:
		if knownNodevalue[0]==0:
			dummyVector[:2]+=array([f[0],c[0]])*knownNodevalue[1]
			c=delete(c,0);f=delete(f,0);b=delete(b,0)
		if knownNodevalue[0]==1:
			dummyVector[-2:]+=array([ b[-1],f[-1]])*knownNodevalue[1]
			c=delete(c,-1);f=delete(f,-1);b=delete(b,-1)
	RHS_primary=RHS-dummyVector.reshape(nElements+1,1)
	RHS_primary=delete(RHS_primary,sec_loc,0)
	K_primary=array([c,f,b])
	secondaryVariables=array([])
	return K_primary,RHS_primary,secondaryVariables
	
def SolverinitializeFULLmatrix(K_global,RHS,essentialBC,sec_loc):
	K_primary=K_global.copy()
	RHS_primary=RHS.copy()
	nElements=(K_global[0].size)-1
	#we want to solve primary variables first and we know the values at ESSENTIAL BC conditions
	#so we remove those
	for knownNodevalue in essentialBC:
		BC_node_index=knownNodevalue[0]*nElements
		K_primary=delete(K_primary,BC_node_index,1)
		RHS_primary=RHS-(knownNodevalue[1]*K_global[:,BC_node_index]).reshape(nElements+1,1)
	#wa cant utilize RHS values at locations where there are secondary terms at play
	secondaryVariables=array([])
	K_primary=delete(K_primary,sec_loc,0)
	RHS_primary=delete(RHS_primary,sec_loc,0)
	if sec_loc: secondaryVariables=RHS[sec_loc,:]
	return K_primary,RHS_primary,secondaryVariables

def meshrefinementDiscountinous(oldMesh,oldSolution,dsc_index,vars,MeshMultiplier,MeshRefinementParameter=0.5,verbose=False):
	Mesh1=oldMesh[:dsc_index+1];       Mesh2=oldMesh[dsc_index:]
	ux_g_t1=oldSolution[:dsc_index+1]; ux_g_t2=oldSolution[dsc_index:]	
	
	newNodes1=(meshrefinement(Mesh1,ux_g_t1,MeshMultiplier,MeshRefinementParameter,verbose=False))[:-1]
	newNodes2=meshrefinement(Mesh2,ux_g_t2,MeshMultiplier,MeshRefinementParameter,verbose=False)
	NewMesh=append(newNodes1,newNodes2)
	if verbose:
		y1=oldSolution[:-2];y2=oldSolution[1:-1];y3=oldSolution[2:]
		x1=oldMesh[:-2];x2=oldMesh[1:-1];x3=oldMesh[2:]
		u_secondDerivative=2*( y1/((x2-x1)*(x3-x1))-y2/((x3-x2)*(x2-x1))+y3/((x3-x2)*(x3-x1)))
		figure();plot(oldMesh[1:-1],abs(u_secondDerivative));title('SECOND DERIVATIVES');show()
		figure();bar(NewMesh[1:],(NewMesh[1:]-NewMesh[:-1]),width=((NewMesh[1:]-NewMesh[:-1])*1.6));title('Element LENGTHS');show()
	
	dsc_index=len(newNodes1)
	if isinstance(vars, np.ndarray):
		DiscVars1=ones_like(newNodes1)*vars[:,:,0]
		DiscVars2=ones_like(newNodes2)*vars[:,:,1]
		discVars=hstack([DiscVars1,DiscVars2[:,:-1]])
	return NewMesh,discVars,dsc_index
	
	
	
def meshrefinement(oldMesh,oldSolution,MeshMultiplier,MeshRefinementParameter=0.5,verbose=False):
	MAGIC_refinement_number=MeshRefinementParameter
	Ux=oldSolution
	xL=oldMesh[0]; xR=oldMesh[-1]
	lengths=oldMesh[1:]-oldMesh[:-1]
	oldmeshsize=oldMesh.size
	
	newmeshsize=MeshMultiplier*(oldmeshsize-1)
	newmeshinput=arange(0,1,1/newmeshsize)
	newmeshinput=newmeshinput/newmeshinput[-1]

	y1=Ux[:-2];y2=Ux[1:-1];y3=Ux[2:]
	x1=oldMesh[:-2];x2=oldMesh[1:-1];x3=oldMesh[2:]
	u_secondDerivative=2*( y1/((x2-x1)*(x3-x1))-y2/((x3-x2)*(x2-x1))+y3/((x3-x2)*(x3-x1)))

	if verbose:figure();plot(abs(u_secondDerivative));plt.title('SECOND DERIVATIVES');show()
	u_secondDerivative=abs(u_secondDerivative)
	meshrefineCriteria=u_secondDerivative/mean(u_secondDerivative)
	meshrefineCriteria=1/meshrefineCriteria ** MAGIC_refinement_number
	meshrefineX=(arange(0,1,1/(meshrefineCriteria.size)));meshrefineX=meshrefineX/meshrefineX[-1]
	if verbose:figure();plot(meshrefineX,meshrefineCriteria);plt.title('new mesh mapping>>>>>>smaller steps in HIGH second derivative regions');show()

	newmeshlengths=interp(newmeshinput,meshrefineX,meshrefineCriteria)
	New_mesh1=cumsum( newmeshlengths )
	New_mesh1-=New_mesh1[0]
	New_mesh1/=New_mesh1[-1]/(xR-xL)
	New_mesh1+=xL
	return New_mesh1
	
def SolverMainAnalyticHW3(k_values,q_values,nodes,naturalBC,essentialBC,tridiag=False,verbose=False):
	lengths=nodes[1:]-nodes[:-1]
	nElements=lengths.size
	R_mean_list=(Mesh[1:]+Mesh[:-1])/2
	#building Elemental Matrices
	Q,sec_loc=build_naturalBC_term(nElements,naturalBC,verbose)
	K_elemental=build_elemental_coefficient_TENSOR_spherical_ANALYTIC(k_values,R_mean_list,lengths,tridiag)
	f_elemental=RHS_elemental_forHW3_analytic(lengths,R_mean_list,q_values)
	#building Global Matrices
	K_global,F_global=build_GLOBAL_Matrices(K_elemental,f_elemental,tridiag)
	RHS=F_global+Q
	#we want to solve primary variables first and we know the values at ESSENTIAL BC conditions
	K_primary,RHS_primary,secondaryVariables=Solverinitialize(K_global,RHS,essentialBC,sec_loc,tridiag)
	Ux=SolverPrimary(K_primary,RHS_primary,essentialBC,tridiag)
	return Ux
	
	
def SolverMainGauss(Shape_Functions,Aprox_Func_partialDE_Matrix,LHS_variable_func,RHS_variable_func,LHS_local_scalar,RHS_local_scalar,nodes,naturalBC,essentialBC,n=2,tridiag=False,verbose=False):
	if len(naturalBC)==1 and len(essentialBC)==1 and naturalBC[0][0]==essentialBC[0][0]: 
		tridiag=False; print('THOMAS SHUT DOWN')
	lengths=nodes[1:]-nodes[:-1]
	nElements=lengths.size
	R_mean_list=(nodes[1:]+nodes[:-1])/2
	jacobian=lengths/2
	LHS_constants=LHS_local_scalar/jacobian
	RHS_element_constants=RHS_local_scalar*jacobian
	
	Q,sec_loc=build_naturalBC_term(nElements,naturalBC,verbose)
	K_elemental=build_elemental_coefficient_TENSOR_GAUSS(LHS_constants,R_mean_list,lengths,LHS_variable_func,Aprox_Func_partialDE_Matrix,n,tridiag)
	f_elemental=RHS_elemental_forHW3_Gauss(RHS_element_constants,R_mean_list,lengths,Shape_Functions,RHS_variable_func,n)
	#building Global Matrices
	K_global,F_global=build_GLOBAL_Matrices(K_elemental,f_elemental,tridiag)
	RHS=F_global+Q
	#we want to solve primary variables first and we know the values at ESSENTIAL BC conditions
	K_primary,RHS_primary,secondaryVariables=Solverinitialize(K_global,RHS,essentialBC,sec_loc,tridiag)
	Ux=SolverPrimary(K_primary,RHS_primary,essentialBC,tridiag)
	return Ux

def aproximationFunctions(psy):  return array([0.5*(1-psy),0.5*(1+psy)])#return matrix of aproximation functions evaluated at apropriate psy legendre points
def elementalVariableFunction_Rsquare(local_pshyi,meanValues,lengths): return (meanValues+(lengths/2)*local_pshyi)**2
#problem definition
'''
r0=0; r_core=0.1/10; r_case=1/10
g0=6*(10**6)
k_core=2.8; k_clad=18
q=[g0,0]; k=[k_core,k_clad]
T_o=420
'''

r0=0; r_core=0.1; r_case=1
g0=15*(10**6)
k_core=20; k_clad=35
q=[g0,0]; k=[k_core,k_clad]
T_o=400

r_analytic_list,T_analytic_list=AnalyticSpherical(r_case,r_core,k_core,k_clad,g0,T_o)


#BC_CONDITIONS 1 means Right end 0 means left
BC_essential1=(1,T_o)
#BC_essential2=(0,50)
BC_natural1=(0,0)
#BC_natural2=(1,20)

essentialBC=[];naturalBC=[]
essentialBC.append(BC_essential1)
#essentialBC.append(BC_essential2)
naturalBC.append(BC_natural1)
#naturalBC.append(BC_natural2)
n_ELEMENTS=10
VARS=array([[q],[k]])

Mesh,disc_variables,dsc_index=MesherConstantDiscontinous(r0,r_core,r_case,n_ELEMENTS,VARS)
q_values=disc_variables[0]; k_values=disc_variables[1]
#Ux_analytic2=SolverMainAnalyticHW3(k_values,q_values,Mesh,naturalBC,essentialBC,tridiag=True)#FIRST SOLUTION checking if analytic aproach is bug free before moving forward for NUMERIC INTEGRATION


def aproximationFunctions(psy):  return array([0.5*(1-psy),0.5*(1+psy)])#return matrix of aproximation functions evaluated at apropriate psy legendre points
def elementalVariableFunction_Rsquare(local_pshyi,meanValues,lengths): return (meanValues+(lengths/2)*local_pshyi)**2
Shape_Functions=aproximationFunctions
Aprox_Func_partialDE_Matrix=array([[1/4,-1/4],[-1/4,1/4]])
LHS_variable_func=elementalVariableFunction_Rsquare
RHS_variable_func=elementalVariableFunction_Rsquare
LHS_local_scalar=k_values
RHS_local_scalar=q_values

#SOLUTION 1 with original MESH 
Ux_1=SolverMainGauss(Shape_Functions,Aprox_Func_partialDE_Matrix,LHS_variable_func,RHS_variable_func,LHS_local_scalar,RHS_local_scalar,Mesh,naturalBC,essentialBC,n=2,tridiag=True)
ERROR,__dummy=getError(r_case,r_core,k_core,k_clad,g0,T_o,Mesh,dsc_index,Ux_1,verbose=True)



#SOLUTION 2 with BETTER ARRANGED MESH
NewMesh,dsc_variables2,dsc_index2=meshrefinementDiscountinous(Mesh,Ux_1,dsc_index,VARS,MeshMultiplier=1.2,MeshRefinementParameter=0.45,verbose=True)
q_values2=dsc_variables2[0]; k_values2=dsc_variables2[1]
LHS_local_scalar2=k_values2
RHS_local_scalar2=q_values2
Ux_New=SolverMainGauss(Shape_Functions,Aprox_Func_partialDE_Matrix,LHS_variable_func,RHS_variable_func,LHS_local_scalar2,RHS_local_scalar2,NewMesh,naturalBC,essentialBC,n=2,tridiag=True)
ERROR2,__dummy=getError(r_case,r_core,k_core,k_clad,g0,T_o,NewMesh,dsc_index2,Ux_New)


#SOLUTION 3 with BETTER ARRANGED reasonably fine MESH
NewMesh2,dsc_variables3,dsc_index3=meshrefinementDiscountinous(Mesh,Ux_1,dsc_index,VARS,MeshMultiplier=2,MeshRefinementParameter=0.25,verbose=False)
q_values3=dsc_variables3[0]; k_values3=dsc_variables3[1]
LHS_local_scalar3=k_values3
RHS_local_scalar3=q_values3
Ux_New2=SolverMainGauss(Shape_Functions,Aprox_Func_partialDE_Matrix,LHS_variable_func,RHS_variable_func,LHS_local_scalar3,RHS_local_scalar3,NewMesh2,naturalBC,essentialBC,n=2,tridiag=True)
ERROR2,__dummy=getError(r_case,r_core,k_core,k_clad,g0,T_o,NewMesh2,dsc_index3,Ux_New2)


#SOLUTION 4 with BETTER ARRANGED super fine MESH
NewMesh3,dsc_variables4,dsc_index4=meshrefinementDiscountinous(Mesh,Ux_1,dsc_index,VARS,MeshMultiplier=2500,MeshRefinementParameter=0.4,verbose=False)
q_values4=dsc_variables4[0]; k_values4=dsc_variables4[1]
LHS_local_scalar4=k_values4
RHS_local_scalar4=q_values4
Ux_New3=SolverMainGauss(Shape_Functions,Aprox_Func_partialDE_Matrix,LHS_variable_func,RHS_variable_func,LHS_local_scalar4,RHS_local_scalar4,NewMesh3,naturalBC,essentialBC,n=2,tridiag=True)
ERROR3,__dummy=getError(r_case,r_core,k_core,k_clad,g0,T_o,NewMesh3,dsc_index4,Ux_New3)


#SOLUTION 5 with RETARDED fine MESH
NewMesh4,dsc_variables5,dsc_index5=meshrefinementDiscountinous(Mesh,Ux_1,dsc_index,VARS,MeshMultiplier=2000000,MeshRefinementParameter=0.4,verbose=False)
q_values5=dsc_variables5[0]; k_values5=dsc_variables5[1]
LHS_local_scalar5=k_values5
RHS_local_scalar5=q_values5
Ux_New4=SolverMainGauss(Shape_Functions,Aprox_Func_partialDE_Matrix,LHS_variable_func,RHS_variable_func,LHS_local_scalar5,RHS_local_scalar5,NewMesh4,naturalBC,essentialBC,n=2,tridiag=True)
ERROR4,__dummy=getError(r_case,r_core,k_core,k_clad,g0,T_o,NewMesh4,dsc_index5,Ux_New4)
