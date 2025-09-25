import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Va=np.array([6,2])
Vb=np.array([3,5])

#Drawing the vectos on a plane
def DrawVector(Va,Vb,T=np.zeros((3,3))):

    if Vb.ndim==1:
        plt.arrow(0, 0, Va[0], Va[1], color=(0, 0, 0), head_width=0.1)
        plt.arrow(0, 0, Vb[0], Vb[1], color=(1, 0, 0), head_width=0.1)
        plt.grid(True)
        plt.show()

    elif np.array_equal(T,np.zeros((3,3))):
        plt.arrow(0, 0, Va[0], Va[1], color=(0, 0, 0), head_width=0.1)
        plt.arrow(0, 0, Vb[0][0], Vb[1][0], color=(1, 0, 0), head_width=0.1)
        plt.grid(True)
        plt.show()
    else:
        plt.arrow(0, 0, Va[0], Va[1], color=(0, 0, 0), head_width=0.1)
        plt.arrow(T[0][2], T[1][2], Vb[0][0]-T[0][2], Vb[1][0]-T[1][2], color=(1, 0, 0), head_width=0.1)
        plt.grid(True)
        plt.show()

DrawVector(Va,Vb)
def Drawparallogram(Va,Vb):
    plt.arrow(0, 0, Va[0], Va[1], color=(0, 0, 0), head_width=0.1)
    plt.arrow(0, 0, Vb[0], Vb[1], color=(1, 0, 0), head_width=0.1)
    plt.arrow(Vb[0], Vb[1], Va[0], Va[1], color=(0, 0, 0), head_width=0.1)
    plt.arrow(Va[0], Va[1], Vb[0], Vb[1], color=(1, 0, 0), head_width=0.1)
    plt.grid(True)
    plt.show()

#Subtraction
ABDifferences,BADifferences=(Va-Vb),(Vb-Va)
print(f"Differences between vectors a-b is :{ABDifferences} and b-a is {BADifferences}\n\n=========")

#Addition
ABsummation=(Va+Vb)
print(f"Summation of vectors a and b which is b and a is :{ABsummation}\n\n=========")

#Angles
VaAngle=float(f"{np.arctan(Va[1]/Va[0]):.2f}")
VbAngle=float(f"{np.arctan(Vb[1]/Vb[0]):.2f}")

print(f"the angles of vector A is {VaAngle*180/np.pi:.2f} and for Vb it's {VbAngle*180/np.pi:.2f}\n\n=========")

#Euclidean norms:
ENormVa=float(f"{np.sqrt(sum(Va*Va)):.2f}")
ENormVb=float(f"{np.sqrt(sum(Vb*Vb)):.2f}")
print(f"The Euclidian norm of Va is {ENormVa}The Euclidian norm of Vb is {ENormVb}\n\n=========")

#dot product:
print(f"the dot product of a and b is {np.dot(Va,Vb)}\n\n=========")
#Dot product drawing
Drawparallogram(Va,Vb)

#cross product
Va=np.array([6,2,0])
Vb=np.array([3,5,0])
Cross=np.linalg.cross(Va,Vb)
print(f"The crossproduct of vector a and vector b is equal to {Cross}\n\n=========")

#cross product drawing
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0,0,0,Va[0],Va[1],Va[2],color=(0,0,0))
ax.quiver(0,0,0,Vb[0],Vb[1],Vb[2],color=(1,0,0))
ax.quiver(0,0,0,Cross[0],Cross[1],Cross[2],color=(0,1,0))
ax.set_xlim(-5,5)
ax.set_ylim(0,5)
ax.set_zlim(0,20)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
plt.show()


#Matrices operations=======================================
#matrices
MatrixA=np.array([[1,2],
                  [2,4]])
MatrixB=np.array([[1,2],
                  [0,4]])
print(f"summation of matrix a and b is {MatrixA+MatrixB}\n\n=========")

#scaled matrix
print(f"multiplication of matrix A by 2 is {2*MatrixA} and B by 3 is {3*MatrixB}\n\n=========")

#Hadamard product
print(f"the Hadamard product is \n{MatrixA*MatrixB}")

#Determinant
print(f"determinant of matrix A is {np.linalg.det(MatrixA)} And matrix B is {np.linalg.det(MatrixB)}\n\n=========")


#Trasformations===============================
Rangle=np.deg2rad(45)
V=np.array([[6],[2],[1]])

S=np.array([[2,0,0],[0,3,0],[0,0,1]])
T=np.array([[1,0,3],[0,1,2],[0,0,1]])
R=np.array([[np.cos(Rangle),-np.sin(Rangle),0],[np.sin(Rangle),np.cos(Rangle),0],[0,0,1]])
# #scaling the vector
SV=np.linalg.matmul(S,V)
print(f"this is the scaled matrix {SV}\n\n=========")
DrawVector(Va,SV)

# #Rotating the vector
RV=np.linalg.matmul(R,V)
print(f"this is the value of the rotated matrix is{RV}\n\n=========")
DrawVector(Va,RV)
# #rotation and scaling

RS=np.linalg.matmul(R,S)
RSV=np.linalg.matmul(RS,V)
DrawVector(Va,RSV)
print(f"this is the value of the rotated and scaled matrix is{RSV}\n\n=========")

# #Translation and rotation and scaling
TRS=np.linalg.matmul(T,RS)
TRSV=np.linalg.matmul(TRS,V)
DrawVector(Va,TRSV)


H=TRS
print(f"the value of the homogenous matrix is {H}\n\n=========")

Vh=np.linalg.matmul(H,V)
print(f"applying the homogenous matrix to a vector {Vh}")