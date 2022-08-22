from pylab import *
import numpy.linalg 
import numpy as np
import matplotlib.pyplot as plt


def MesherConstant(xL,xR,nElements):
	elemLen=(xR-xL)/nElements
	Nodes=arange(xL,xR+elemLen/2,elemLen)
	return Nodes

def build_elemental_coefficient_TENSOR(a,c,element_lengths):
	Hv1=1/(element_lengths); Hv2=element_lengths;
	T1=array([[a,-a],[-a,a]])
	T2=array([[c/3,c/6],[c/6,c/3]])
	TensorT=np.array([T1,T2])
	Tensor_vec=np.array([Hv1,Hv2])
	#builds element Coefficient matrices
	K_elements_TENSOR=tensordot(transpose(Tensor_vec),(TensorT),1)
	return K_elements_TENSOR

def build_GLOBAL_Matrices(K_tensor,f_elemental):
	'''hocam loop kullanmayi kendime gecen yil yasakladigim icin bu fonksiyon var
	direk local elementler K tensorunu Global K matrixine mapliyorum
	'''
	nElements=K_tensor[:,0,0].size
	T_local2Global=generateTransformationTensors(nElements)
	TransT=transpose(T_local2Global)
	rhs=einsum('ijk,kai->ija',K_tensor,T_local2Global)
	K_global=einsum('ijk,ikb->jb',TransT,rhs)
	F_global=einsum('ijk,kfi->jf',TransT,f_elemental)
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

def build_naturalBC_term(nElem,naturalBC,verbose=False):
	Q=np.zeros(nElem+1)
	SecondaryVariableLocations=[0,nElem+1]
	for item in naturalBC:
		Q[item[0]]=item[1]
		if item[0] in SecondaryVariableLocations:
			SecondaryVariableLocations.remove(item[0])
	if verbose:print(f'Q BC vector is\n {Q.reshape(nElem+1,1)}\n secondary variable Locations are \n {SecondaryVariableLocations}\n')
	return Q.reshape(nElem+1,1), SecondaryVariableLocations
	
def SolverPrimary(K_primary,RHS_primary,Essential_BC):
	#U=dot(inv(K_primary),RHS_primary)
	RHS_primary=RHS_primary.flatten()
	U=solve(K_primary,RHS_primary)
	for node in Essential_BC:
		if node[0]==0: U=append(node[1],U)
		else:
			Ubefore=append(U[:node[0]-1],node[1])
			Uafter=U[node[0]-1:]
			U=append(Ubefore,Uafter)
	return U

def Solverinitialize(K_global,RHS,essentialBC,sec_loc):
	K_primary=K_global.copy()
	RHS_primary=RHS.copy()
	nElements=(K_global[0].size)-1
	#we want to solve primary variables first and we know the values at ESSENTIAL BC conditions
	#so we remove those
	for knownNodevalue in essentialBC:
		K_primary=delete(K_primary,knownNodevalue[0],1)
		RHS_primary=RHS-(knownNodevalue[1]*K_global[:,knownNodevalue[0]]).reshape(nElements+1,1)
	#wa cant utilize RHS values at locations where there are secondary terms at play
	secondaryVariables=array([])
	for unKnownBC in sec_loc:
		K_primary=delete(K_primary,unKnownBC-1,0)
		RHS_primary=delete(RHS_primary,unKnownBC-1,0)
		secondaryVariables=append(secondaryVariables,RHS[unKnownBC-1])
	return K_primary,RHS_primary,secondaryVariables
	
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
	return New_mesh1
	
def SolverMain(a,c,nodes,naturalBC,essentialBC,verbose=False):
	lengths=nodes[1:]-nodes[:-1]
	nElements=lengths.size
	Q,sec_loc=build_naturalBC_term(nElements,naturalBC,verbose)
	K_elemental=build_elemental_coefficient_TENSOR(a,c,lengths)
	f_elemental=RHS_elemental_forHW1(nodes,lengths)
	#building Global Matrices
	K_global,F_global=build_GLOBAL_Matrices(K_elemental,f_elemental)
	RHS=F_global+Q
	#we want to solve primary variables first and we know the values at ESSENTIAL BC conditions
	K_primary,RHS_primary,secondaryVariables=Solverinitialize(K_global,RHS,essentialBC,sec_loc)
	Ux=SolverPrimary(K_primary,RHS_primary,essentialBC)
	fluxRowvector=K_global[sec_loc[0]-1,:]
	
	Flux=dot(fluxRowvector,Ux)-RHS[sec_loc[0]-1]
	return Ux,Flux
	

xL = 0;xR = pi
#weak form DE in the form a*(du/dx2)+c*u=f
#form tensors for the problem using D.E. properties
a=-1; c=4
#BC_CONDITIONS
BC_essential1=(0,0)
#BC_natural1=(0,1)
BC_natural1=(0,0)
essentialBC=[]
naturalBC=[]

essentialBC.append(BC_essential1)
naturalBC.append(BC_natural1)

nElementsT = 10000;nodesT=MesherConstant(xL,xR,nElementsT)
true_solution=cos(2*nodesT)/4-cos(4*nodesT)/4#TRUE ANALYTIC
#true_solution=cos(2*nodesT)/4-cos(4*nodesT)/4+sin(2*nodesT)/2#TRUE ANALYTic for q=1
nElements = 80; nodes=MesherConstant(xL,xR,nElements)#first mesh
Ux,flux=SolverMain(a,c,nodes,naturalBC,essentialBC)

newNodes=meshrefinement(oldMesh=nodes,oldSolution=Ux,MeshMultiplier=0.3,MeshRefinementParameter=0.4,verbose=True)
Ux2,flux2=SolverMain(a,c,newNodes,naturalBC,essentialBC)


nElements=len(newNodes)
nodes3=MesherConstant(xL,xR,nElements)#first mesh
Ux3,flux3=SolverMain(a,c,nodes3,naturalBC,essentialBC,True)

#just checking if it works
nElements = 1000; nodes4=MesherConstant(xL,xR,nElements)#first mesh
Ux4,flux4=SolverMain(a,c,nodes4,naturalBC,essentialBC)

figure()
newlengths=newNodes[1:]-newNodes[:-1]
plot(newlengths)
xlabel('Element NO')
ylabel('Element SIZE')
plt.title(f'Varying element lengths for {newlengths.size} ELEMENTS')
show()

figure()
plot(nodesT,true_solution,linewidth=3,linestyle='--',color="black")
plot(nodes,Ux,color="blue")
plot(newNodes,Ux2,color="green")
plot(nodes3,Ux3,color="red")
xlabel('U')
ylabel('x')
title('U vs x')
plt.xlim(xL,xR)
plt.legend(['ANALYTIC SOLUTION',f'{nodes.size-1} elements with equal steps',(f'{newNodes.size-1} elements with cleverly varying steps'),(f'{nodes3.size-1} elements with equal steps')],loc=3, prop={'size': 8})
show()

figure()
plot(nodesT,true_solution,linewidth=3,linestyle='--',color="black")
plot(nodes,Ux,color="blue")
plot(newNodes,Ux2,color="green")
plot(nodes3,Ux3,color="red")
xlabel('U')
ylabel('x')
plt.xlim(0.5,1.2)
plt.ylim(0,0.8)
title('U vs x')
plt.legend(['ANALYTIC SOLUTION',f'{nodes.size-1} elements with equal steps',(f'{newNodes.size-1} elements with cleverly varying steps'),(f'{nodes3.size-1} elements with equal steps')],loc=3, prop={'size': 8})
show()



figure()
plot(nodes4,Ux4)
show()


#with open('test.npy', 'rb') as f:
#...     a = np.load(f)
#...     b = np.load(f)
