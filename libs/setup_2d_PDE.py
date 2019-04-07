import sys, math
import numpy as np
import matrix_solver as ms

#TODO: Put together stuff for a config file. But for now, we can just use the defaults



def getProblem(filename):
    '''Given a filename string, return various problem configuration options from that file in a dict.'''
    with open(filename) as file: #when you use a with statement to open a file, you don't have to close it again, even if the program throws an error
        variables = {}
        for line in file:
            line = line.strip().rstrip() #get rid of hidden whitespace
            if line and not line.startswith('#'):
                varname,value = line.split(':',2)[0:2]
                varname = varname.strip()
                variables[varname] = value
    #clean up a little by replacing references with the referenced value
        for key in variables.keys():
            values = variables[key].split(',')
            for elem in values:
                elem = elem.strip()
                if elem in variables.keys():
                    newelem = variables[elem]
                    variables[key] = variables[key].replace(elem, newelem)

        for key in variables.keys():
            values = variables[key].split(',')
            i = 0 #index for elements, because I can't figure another way on the fly
            for elem in values:
                elem = elem.strip()
                try:
                    elem = float(elem)
                    
                except ValueError:
                    pass #not able to make it a float, because it's one of the strings that should be there
                values[i] = elem
                i+=1
            variables[key] = values
            

        return variables

                    
    
#throw into config file
##MIN_X = 0 
##MIN_Y = 0
##MAX_X = 3
##MAX_Y = 3
##DX = 0.1
##DY = 0.1
##Q = 1


##
##
###boundary conditions
##TOP = (MIN_X, Q, 'dirichlet') #takes the form MIN TO START, VALUE OF BOUNDARY, TYPE OF CONDITION; takes multiple definitions in one tuple
##BOTTOM = (MIN_X, 0, 'dirichlet')
##LEFT = (MIN_Y, 0, 'neumann')
##RIGHT = (MIN_Y, 0, 'dirichlet', 1, 0, 'neumann', 2, Q, 'dirichlet')
##
##
###cartesian vectors spanning the problem
##NX = math.ceil((MAX_X-MIN_X)/DX) -1
##X = [i*DX for i in range(NX+2)]
##
##NY = math.ceil((MAX_Y-MIN_Y)/DY) -1
##Y = [i*DY for i in range(NY+2)]

#key: phi1 = (1,1), phi2 = (1,2), phi3 = (2,1), phi4 = (2,2) (assuming delta = 1)

#this means, if i is the x index, and y is the j index, and k is the phi index,
#basically, we should insert a value into our phi vector whenever we pass ny values in a loop, assuming we want to roll it up vertically

def findBound(pos, boundary):
    '''Given a boundary tuple and an x or y position, return which condition actually applies
    pos is a position in x or y
    boundary is a tuple in the form (MIN, BOUNDARY_VALUE, TYPE), that can hold any number of boundary conditions
        MIN is the minimum pos where the condition applies
        BOUNDARY_VALUE is the actual value of the boundary condition
        TYPE is the type of boundary condition, either 'dirichlet' or 'neumann'
    returns a tuple in the form of (VAL, TYPE) where VAL is the value of the boundary condition and TYPE is the type, either neumann or dirichlet'''

    for i in range(len(boundary)-3,-1,-3): #go through the boundary tuple backward
        if pos < boundary[0]:
            pos = boundary[0] #catch rounding errors
        if pos >= boundary[i]:
            return(boundary[i+1], boundary[i+2])
        

def buildCartStencil(nx,ny):
    '''Given spatial configurations and boundary conditions for a two-dimensional problem,
return the stencil matrix and vector using a 2nd-order finite difference cartesian stencil.
nx: the number of steps in x across the problem space.
ny: the number of steps in y across the problem space.
top: a tuple of top boundary conditions, in the form of (minbound,maxbound,val,type). Takes multiple bounds, specified sequentially in the same tuple.
    minbound and maxbound specify the minimum and maximum values for which this condition is true
    val is the value at the boundary
    type is either 'd' or 'n' for dirichlet or neumann. Neumann boundaries will be solved through forward Euler, and so will be first-order
bottom, left, right: like top, for the bottom, left, and right-side respectively

    NOTE: This function doesn't contain error checking. Everything must be formatted correctly. Error checking is out of scope'''
    #requires globals
        
    
    #preallocate the matrix and vector, so we'll know early if memory will be a problem
    #A is always square
    #b is always as many rows as A has columns
    

    #assume all boundaries are specified; then there are (nx-1)*(ny-1) points

    size = nx*ny
    A = [[0]*size for i in range(size)]
    b = [[RHO] for i in range(size)]


    #build A and b based on the boundary conditions
    #THIS ROLLS SPACE UP HORIZONTALLY
    #I mean, it advances along x, and then goes to a new y

    x = MIN_X #x position
    y = MIN_Y #y position
    
    k = 1 #our nk, against which we will check to see if we're crossing the boundaries. Also, our column index
    j = 1
    for i in range(size): 
        A[i][i] -= 4 #this should never fail
        x+=DX
        if x >=MAX_X:
            x = MIN_X
            y+=DY
        
        #try x first 
        if j == 1: #apply left boundary for -x
            pos = y
            value, kind = findBound(pos, LEFT)
            if kind == 'dirichlet':
                b[i][0] -= value #set the dirichlet condition in the resultant vector
            elif kind == 'neumann':
                A[i][i] += 1 #using the first order approximation because the 2nd-order approximation involves hoping the matrix is big enough 
                b[i][0] -= DX*value
            A[i][i+1] +=1 #left boundary, so right is free
        elif j == nx: #apply the right boundary for +x
            pos = y
            value, kind = findBound(pos, RIGHT)
            if kind == 'dirichlet':
                b[i][0] -= value #set the dirichlet condition in the resultant vector
            elif kind == 'neumann':
                A[i][i] += 1 #using the first order approximation because the 2nd-order approximation involves hoping the matrix is big enough 
                b[i][0] += DX*value
            A[i][i-1] +=1 #right boundary, so left is free
        else: #next x value does not touch boundary
            A[i][i+1] +=1
            A[i][i-1] +=1

        #now test for +- y
        if k == 1: #apply bottom boundary for -y.
            pos = x
            value, kind = findBound(pos, BOTTOM)
            if kind == 'dirichlet':
                b[i][0] -= value #set the dirichlet condition in the resultant vector
            elif kind == 'neumann':
                A[i][i] += 1 #using the first order approximation because the 2nd-order approximation involves hoping the matrix is big enough 
                b[i][0] += DY*value #toggle to - if this doens't work
            A[i][i+nx] += 1 #bottom boundary, so top is free
        elif k == ny:#apply the top boundary for +y
            pos = x
            value, kind = findBound(pos, TOP)
            if kind == 'dirichlet':
                b[i][0] -= value #set the dirichlet condition in the resultant vector
            elif kind == 'neumann':
                A[i][i] += 1 #using the first order approximation because the 2nd-order approximation involves hoping the matrix is big enough 
                b[i][0] -= DY*value
            A[i][i-nx] += 1 #top boundary, so bottom is free
        else: #next y value does not touch boundary
            A[i][i+nx] +=1
            A[i][i-nx] +=1

        
        if j == nx: #wrap around to +y
            j = 1    
            k +=1

        else:
            j+=1


##  kept for posterity. Let this be a lesson in overcomplicating your problem
##
##        #now figure out where we are in x and y so we can apply the proper boundary conditions
##        
##
##        #set up the A matrix according to the stencil
##
##
##        try: A[i][i-1] += 1
##        except IndexError: #if this fails, the left boundary condition applies
##            pos = x-DX
##            value, kind = findBound(pos, LEFT)
##            if kind == 'dirichlet':
##                b[i][0] -= value #set the dirichlet condition in the resultant vector
##            elif kind == 'neumann':
##                A[i][i] += 1 #using the first order approximation because the 2nd-order approximation involves hoping the matrix is big enough 
##                b[i][0] -= DX*value
##            else:
##                print('Something went seriously wrong in the boundary value definitions. Check them and try again.')
##                raise IndexError
##
##        try: A[i][i+1] += 1
##        except IndexError: #if this fails, the right boundary condition applies
##            pos = x+DX
##            value, kind = findBound(pos, RIGHT)
##            if kind == 'dirichlet':
##                b[i][0] -= value #set the dirichlet condition in the resultant vector
##            elif kind == 'neumann':
##                A[i][i] += 1 #using the first order approximation because the 2nd-order approximation involves hoping the matrix is big enough 
##                b[i][0] -= DX*value
##            else:
##                print('Something went seriously wrong in the boundary value definitions. Check them and try again.')
##                raise IndexError
##
##        try: A[i][i-ny] += 1
##        except IndexError: #if this fails, the bottom boundary condition applies
##            pos = y-DY
##            value, kind = findBound(pos, BOTTOM)
##            if kind == 'dirichlet':
##                b[i][0] -= value #set the dirichlet condition in the resultant vector
##            elif kind == 'neumann':
##                A[i][i] += 1 #using the first order approximation because the 2nd-order approximation involves hoping the matrix is big enough 
##                b[i][0] -= DY*value
##            else:
##                print('Something went seriously wrong in the boundary value definitions. Check them and try again.')
##                raise IndexError
##
##        try: A[i][i+ny] += 1
##        except IndexError: #if this fails, the top boundary condition applies
##            pos = y+DY
##            value, kind = findBound(pos, TOP)
##            if kind == 'dirichlet':
##                b[i][0] -= value #set the dirichlet condition in the resultant vector
##            elif kind == 'neumann':
##                A[i][i] += 1 #using the first order approximation because the 2nd-order approximation involves hoping the matrix is big enough 
##                b[i][0] -= DY*value
##            else:
##                print('Something went seriously wrong in the boundary value definitions. Check them and try again.')
##                raise IndexError
##
##        x += DX #update positions for the next run
##        y += DY
    return (A,b)

def solvePsi(A, b, solver, tol=1e-8, maxiters=1e6):
    '''Given a matrix A and a vector b for the matrix equation A(psi) = b, solve for psi
using one of three different solvers.
            solver: a string telling us what solver to use. Valid values are cg, ji, and np.
                    Note that, right now, only np works properly.
            tol: the tolerance the solver will use. Ignored by the np solver. Defaults to 1e-8
            maxiters: The maximum number of iterations for the solver to run. Ignored by ji and np. Defaults to 1e6'''
            
    if solver == 'cg':
        psi = ms.conjGradSolve(A,b, tol, maxiters)

    if solver == 'ji':
        psi = ms.jacobiSolve(A,b, tol)

    if solver == 'np':
        A = np.matrix(A)
        b = np.matrix(b)
        psi = np.linalg.solve(A,b).tolist()

    return psi
    

def wrapPsi(psi):
    '''Given a vector psi that represents point solutions to the Laplacian defined by
a boundary problem, turn it into a matrix for that space.'''
        
    #requires globals

    #first, wrap up the vector to represent the space of the problem
    i = 0
    row = []
    space = []
    for elem in psi:
        row.append(elem[0])
        i+=1
        if i == NX: #we've gone through all the X values for this row, so time to start another
            space.append(row)
            i = 0
            row = []

    psi = space

    #now that we've made psi a matrix again, pad it out with the boundary values
    left = []
    right = []
    top = []
    bottom = []
    corners = [0,0,0,0] #we'll fill out the corners last of all

    #neumann approx. are first order
    pos = MIN_X #bottom first
    for k in range(NX):
        value, kind = findBound(pos,BOTTOM)
        if kind == 'dirichlet':
            elem = value
        elif kind == 'neumann':
            elem = psi[0][k] - DY*value #from point (x,y+DY)
        else:
            raise RuntimeError("Something went seriously wrong with assigning boundary values for the bottom. Check for typos in boundary definitions.")
        bottom.append(elem)
        pos += DX
        
    pos = MIN_X #top
    for k in range(NX):
        value, kind = findBound(pos,TOP)
        if kind == 'dirichlet':
            elem = value
        elif kind == 'neumann':
            elem = psi[-1][k] + DY*value #from point (x,y-DY)
        else:
            raise RuntimeError("Something went seriously wrong with assigning boundary values for the top. Check for typos in boundary definitions.")
        top.append(elem)
        pos +=DX
        
    pos = MIN_Y #left
    for k in range(NY):
        value, kind = findBound(pos,LEFT)
        if kind == 'dirichlet':
            elem = value
        elif kind == 'neumann':
            elem = psi[k][0] - DX*value #from point (x+DX,y)
        else:
            raise RuntimeError("Something went seriously wrong with assigning boundary values for the left. Check for typos in boundary definitions.")
        left.append(elem)
        pos +=DY

    pos = MIN_Y #right
    for k in range(NY):
        value, kind = findBound(pos,RIGHT)
        if kind == 'dirichlet':
            elem = value
        elif kind == 'neumann':
            elem = psi[k][-1] + DX*value #from point (x-DX, y)
        else:
            raise RuntimeError("Something went seriously wrong with assigning boundary values for the right. Check for typos in boundary definitions.")
        right.append(elem)
        pos +=DY

    #now we do the corners, which require special handling. Fortunately, we know there are only four of them
    #I could easily just shove these into the other lists, but this makes for easier reading
    corners[0] = (left[0] + bottom[0])/2 #lower left. Average of two nearest points, because that's less intensive and roughly equivalent to averaging the two different ways of getting that point.
    corners[1] = (right[0] + bottom[0])/2 #lower right
    corners[2] = (left[-1] + top[0])/2 #upper left
    corners[3] = (right[-1] + top[-1])/2 #upper left
    
    #and now that we've defined our boundary lists, actually append them
    #bottom
    bottom.insert(0,corners[0])
    bottom.append(corners[1])
    psi.insert(0,bottom)
    

    #top (these two are easiest)
    top.insert(0,corners[2])
    top.append(corners[3])
    psi.append(top)
    

    #left
    row = 1
    for elem in left:
        psi[row].insert(0, elem)
        row+=1

    #right
    #could have done this together with left, but using 'for elem' instead of 'for index' is more readable
    #so I sacrifice a little efficiency in calling two for-loops
    row = NY
    for elem in right:
        psi[row].append(elem)
        row-=1

        
    return psi


def yComp(psi, i, j, dx):
    '''Given a matrix psi that represents the point solutions to some function psi(x,y),
    and both a row and column index, return the point solution
    V = -dpsi/dx, the y-component of the vector field at that point
    dx:    the step size in x'''

    V = -(psi[i][j+1] - psi[i][j-1])/(2*dx)
    return V

def xComp(psi, i, j, dy):
    '''Given a matrix psi that represents the point solutions to some function psi(x,y),
    and both a row and column index, return the point solution
    U = dpsi/dy, the x-component of the vector field at that point
    dy:    the step size in y'''
    U = (psi[i+1][j] - psi[i-1][j])/(2*dy)
    return U

def vectorField(psi,dx,dy):
    '''Given a matrix psi that represents the point solutions to some function psi(x,y), return
a matrix that represents the vector field for that function.
        Note that the returned matrix will be 1 smaller in both dimensions than psi
        dx: the step size in x of the psi matrix
        dy; the step size in y of the psi matrix'''
    m = len(psi) - 2
    n = len(psi[0]) - 2
    U = [[0 for i in range(n)] for j in range(m)]
    V = [[0 for i in range(n)] for j in range(m)]
    
    for i in range(1,m):
        for j in range(1,n):
            U[i][j] = xComp(psi, i, j, dy)
            V[i][j] = yComp(psi, i, j, dx)
            
    
    return (U,V)


#down here at the bottom we're going to set up the module-level variables

VARIABLES = getProblem('problem.cfg')

#x and y config
MIN_X = VARIABLES['min_x'][0]
MIN_Y = VARIABLES['min_y'][0]
MAX_X = VARIABLES['max_x'][0]
MAX_Y = VARIABLES['max_y'][0]
DX = VARIABLES['dx'][0]
DY = VARIABLES['dy'][0]
NX = math.ceil((MAX_X-MIN_X)/DX) -1
X = [i*DX for i in range(NX+2)]
NY = math.ceil((MAX_Y-MIN_Y)/DY) -1
Y = [i*DY for i in range(NY+2)]

#boundaries
TOP = VARIABLES['TOP']
BOTTOM = VARIABLES['BOTTOM']
RIGHT = VARIABLES['RIGHT']
LEFT = VARIABLES['LEFT']
Q = VARIABLES['Q'][0] #changing this won't do you any good; it's only used for the graphing bit
RHO = VARIABLES['rho'][0]

#how will we solve?
SOLVER = VARIABLES['solver'][0]
TOL = VARIABLES['tol'][0]
MAXITERS = VARIABLES['maxiters'][0]
