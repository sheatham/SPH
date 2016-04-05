#!/usr/bin/env python
#!/usr/bin/env python

#1D SPH code
#equations from Monaghan 2005

import sys
import numpy as np
import matplotlib.pyplot as mpl

##########################
#init conditions & params#
##########################

n = 10 #sqrt num particles

#length
maxx = 5
minx = -maxx

#courant safety factor
#csf= 0.05 #variable? up to 0.3. for var timestep

vx = np.zeros(n*n)+30
vx = np.array([vx.flatten()]).T
vy = np.zeros(n*n)+30
vy = np.array([vy.flatten()]).T

vhx = np.zeros(n*n) #v at half time step
vhx = np.array([vhx.flatten()]).T
vhy = np.zeros(n*n)
vhy = np.array([vhy.flatten()]).T

xr0 = np.linspace(minx,maxx,n) #init particle spacing (even)
yr0 = np.linspace(minx,maxx,n)
xr, yr = np.meshgrid(xr0,yr0)
xr, yr = np.array([xr.flatten()]).T, np.array([yr.flatten()]).T
#r=np.array([xr.flatten(),yr.flatten()]).T

mag=np.zeros(n*n,dtype=int) #magnitude of r vector

#init analytical densities
rho = np.ones(n*n)
rho = np.array([rho.flatten()]).T
#rho[0:30] = 15

dx = xr[1]-xr[0] #init space between two particles
dx = dx[0]
h = dx*2.0 #smoothing length; decrease with fewer particles
m = dx*rho[:] #mass per particle from density

#init pressure

k=1.0
P = k*rho

########################################
#specific internal energy for ideal gas law formulation of pressure
########################################

#eps=np.zeros(n*n)
#should there be an initial condition for energy?

#gamma = 1.4

#Alternate pressure equation from ideal gas law
#P = (gamma-1.0)*rho*eps
########################################
########################################


#speed of sound
#cs = sqrt(Gamma*(P+B)/rho)
cs=1/rho
#cs=sqrt(Gamma*B/rho)

#File location parameters
parentDir='/Users/sheacheatham/Research/2DSPH/run2/'
imgDir='2DSPH'


# making the neighbs optimization grid. creates empty lists in which particles will be put.
#length = int((abs(maxx)+abs(minx)))
#for i in range(0,maxx):
#    for j in range(0,maxx):
#        gridname = "" #clear out the previous command
#        gridname = "gridnum" + str(i) + str(j) + " = []"
#        globals()[gridname] = []
  
##################
#define functions#
##################

#smoothing kernel: cubic spline
def kernel(rij,h):
    q = rij/h
    D1 = 15./14.
    hsq= h*h
    D = D1/ pi/ hsq
    if q >= 0.0 and q < 1.0:
        return D*(4.0 - 6 * q**2 + 3*q**3)
    elif q < 2.0:
        return D*(2.0-q)**3
    else:
        return 0.0
    
#kernel gradient (GradjWij) 
def gradkernel(rijx,rijy,h):
    u = np.fabs(rijx) / h
    v = np.fabs(rijy) / h
    h3 = h*h*h
    invh = 1./h3
    D1 = 15./14.
    Ddr = D1 / h3*pi
    dxf = 0
    dyf = 0
    if u >= 0.0 and u < 1.0:
        gradx = invh*Ddr*3.0*rijx*(3*sqrt(rijx**2+rijy**2) -4*h)
        dxf=gradx
    elif u < 2.0:
        gradx = invh*Ddr*-3.0*rijx*(sqrt(rijx**2+rijy**2) - 2.0*h)**2 / (h**3 * sqrt(rijx**2+rijy**2))
        dxf=gradx
    else:
        dxf=0.0
    if v >= 0.0 and v < 1.0:
        grady = invh*Ddr*3.0*rijy*(3*sqrt(rijx**2+rijy**2) -4*h)
        dyf=grady
    elif v < 2.0:
        grady = invh*Ddr*-3.0*rijy*(sqrt(rijx**2+rijy**2) - 2.0*h)**2 / (h**3 * sqrt(rijx**2+rijy**2))
        dyf=grady
    else:
        dyf=0.0
        
    return dxf,dyf

#search neighbors (for every particle i, find neighbors closer than 2h)
def find_neighbors(xr,yr,h,n):
    neighbs = np.zeros(n*n,dtype=int)
    #will give num of neighbors for each particle i,
    #including self. Think of it like the num particles in a section of a grid.
    
    neighb_loc = np.zeros( (n*n,n*n), dtype=int)
    #creates a matrix i1[j1 j2] identifying which particle j's affect any give particle i.
    #                 i2[j1 j2]
  
#this is the approach for creating a fixed grid and only checking neighbor grids   
    boxes = loc_particles(xr,yr);
    top, bottom, left, right = edge_boxes()
    checkboxes = []
    num = (maxx+abs(minx))
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            for k in range(len(boxes[i])): #found particle in box
                
            
            neighborBoxes = frozenset([i-(num+1),i-num,i-(num-1),i-1,i,i+1,i+(num-1),i+num,i+(num+1)])#all specific to maxxminset
            if j in neighborBoxes:
                checkboxes.append(j)#now we have our neighboring boxes
            for j in range(len(boxes[i])): #now we have our particle in question
                boxes[i][j]
            
            
            
 
#this is the brute force approach    
#    for i in range(n*n):
#        for j in range(n*n):
#            dist=sqrt((xr[i]-xr[j])**2+(yr[i]-yr[j])**2)
#            if dist < 2.0*h or np.fabs(xr[i]-xr[j]) < 2.0*h: 
#                neighb_loc[i,neighbs[i]] = j
#                neighbs[i]+=1
#            elif np.fabs(yr[i]-yr[j]) < 2.0*h:
#                neighb_loc[i,neighbs[i]] = j
#                neighbs[i]+=1
#                
#    return(neighbs,neighb_loc)

#make the grid
def makeboxes():
    num = (maxx+abs(minx))
    boxes = [ [] for n in range(num*num) ]
    return boxes

def edge_boxes(): #gives box numbers for edge boxes
    top = []
    bottom = []
    left = []
    right = []
    for k in range(1,num):
        tops = 0 + k
        bottoms = num**2-(num-k)
        lefts = num*k
        rights = num*k - 1
        top.append(top)
        bottom.append(bottom)
        left.append(left)
        right.append(right)
    return top, bottom, left, right

#determine which box each particle should be put in
def loc_particles(xr,yr):
    boxes = makeboxes()
    for i in range(len(xr)):
        xkey = int(floor(xr[i]))
        ykey = int(floor(yr[i]))
        if xkey >= maxx:
            xkey = maxx-1
        elif xkey <= minx:
            xkey = minx+1
        if ykey >= maxx:
            ykey = maxx-1
        elif ykey <= minx:
            ykey = minx+1
        box = xkey- 10*ykey + 45 #this is intimately related to the maxx and minx values. don't mess with them.
        position=xr[i],yr[i]
        boxes[box].append(position)
    return  boxes


    

#get smoothed density
def get_density(xr,yr,h,n,m,neighbs,neighb_loc):
    
    avrho=np.zeros(n*n) 
    for i in range(n*n):
        #for j in range(n):
        for j in range(neighbs[i]):
            rij = sqrt((xr[neighb_loc[i,j]]-xr[i])**2+(yr[neighb_loc[i,j]]-yr[i])**2)
            avrho[i]+= m[neighb_loc[i,j]]*kernel(rij,h)
    return avrho

#get artificial viscosity for each paricle i w
#respect to each of its neighbors j
def get_visc(xr,yr,h,vx,vy,rho,cs,neighbs,neighb_loc):
    
    av = np.zeros((n*n,n*n))  
    #viscous constants
    alpha = 1.0
    beta = 2.0    
    for i in range(n*n):
        #for j in range(n):
        for j in range(neighbs[i]):
            vij = sqrt((vx[i]-vx[neighb_loc[i,j]])**2 + (vy[i]-vy[neighb_loc[i,j]])**2) #make sure
            rij = sqrt((xr[neighb_loc[i,j]]-xr[i])**2+(yr[neighb_loc[i,j]]-yr[i])**2)
            #print vij, rij
            if (vij*rij < 0.):
                mew = h * vij*rij/(rij**2 + (0.1*h)**2)
                cij = 0.5*(cs[i]+cs[neighb_loc[i,j]]) #mean speed of sound, this should be fine
                rhoij = 0.5*(rho[i]+rho[neighb_loc[i,j]]) #mean density for i & j, should also be fine
                av[i,neighb_loc[i,j]] = (-alpha*cij*mew + beta*mew**2)/rhoij   
    return av

#get acceleration for each particle
def get_accel(n,neighbs,neighb_loc,press,rho,m,xr,yr,h):   
    xaccels = np.zeros(n*n)
    yaccels= np.zeros(n*n)   
    for i in range(n*n):
        for j in range(neighbs[i]):
            #rij = sqrt((xr[neighb_loc[i,j]]-xr[i])**2+(yr[neighb_loc[i,j]]-yr[i])**2)
            rijx = xr[neighb_loc[i,j]]-xr[i]
            rijy = yr[neighb_loc[i,j]]-yr[i] #not sure about this
            k = neighb_loc[i,j]
            kernx, kerny = gradkernel(rijx,rijy,h)
            xaccels[i] -= m[k] * (press[i]/rho[i]**2 + press[k]/rho[k]**2 + av[i,k]) * kernx
            yaccels[i] -= m[k] * (press[i]/rho[i]**2 + press[k]/rho[k]**2 + av[i,k]) * kerny
            #print(i,j,accels[i])
    return xaccels, yaccels
    
#get energy/dt
#def get_energy(n,neighbs,neighb_loc,v,xr,yr,p,rho,av,h):  
#    energies=np.zeros(n)   
#    for i in range(n):
#        #for j in range(n):
#        for j in range(neighbs[i]):
#            k=neighb_loc[i,j]
#            vij = v[i]-v[k]
#            rij=r[i]-r[k]
#            #energies[i] += (p[i]/rho[i])*(m[k]/rho[k])*vij*drkernel(rij,h)
#            energies[i] += 0.5*m[k]*((p[i]/rho[i]**2)+(p[k]/rho[k]**2)+av[i,k])*vij*drkernel(rij,h)           
#    return energies

#adjusting timestep
#def dts(csf,h,v,accels,cs):
#    odt = 1.0e0 #orig timestep
    #dt = csf*h/(max(1,max(v)))
    #if sqrt(h/max(np.fabs(accels))) < dt:
    #    dt = sqrt(h/max(np.fabs(accels)))
    #else:
    #    if csf*h/max(cs) < dt:
    #        dt = csf*h/max(cs)
    #    else:
    #        if min(csf*h/max(cs)) < dt:
    #            dt = csf*h/max(cs)
    #        else:
    #            dt=dt  
    #return abs(dt)
    #dt = csf*h/max(cs)
    #return np.fmin(1.1*odt,dt)
#    return 0

#saving images
def Output(j,rho,xr,yr,t):
    print(j,t,dt)
    mpl.clf()
    mpl.plot(xr,yr,".")
    mpl.axis([-16,16,-16,16])
    pause(0.0001)
    if j==1 or j % 1 == 0:
        jstr=str(j);
        while len(jstr)<4:
            jstr='0'+jstr
        imgfile=parentDir+imgDir+'img'+jstr+'.png'
        savefig(imgfile)
        
def boundf(xaccels,yaccels,xr,yr):
    kc = 300 #spring constant force for wall. The larger this number, the smaller the timestep has to be.
    for i in range(len(xaccels)):
        if (xr[i] > maxx):
            xaccels[i]=xaccels[i]-kc*abs(xr[i])
        elif (xr[i] < minx):
            xaccels[i]=xaccels[i]+kc*abs(xr[i])
        else:
            xr[i]=xr[i]
        if (yr[i] > maxx):
            yaccels[i]=yaccels[i]-kc*abs(yr[i])
        elif (yr[i] < minx):
            yaccels[i]=yaccels[i]+kc*abs(yr[i])
            #print yaccels[i],yr[i]
        else:
            yr[i]=yr[i]
    return xaccels, yaccels

#cylindrical boundary

def cylboundf(xaccels,yaccels,xr,yr):
    kc = 300
    for i in range(len(xr)):
        if xr[i]**2+yr[i]**2 > maxx**2:
            if xr[i] < 0:
                xaccels[i]=xaccels[i]+kc*abs(xr[i])
            elif xr[i] > 0:
                xaccels[i]=xaccels[i]-kc*xr[i]
            if yr[i] < 0 :
                yaccels[i] = yaccels[i]+kc*abs(yr[i])
            elif yr[i] > 0:
                yaccels[i]=yaccels[i]-kc*yr[i]
    return xaccels, yaccels


def f(xr,yr,vx,vy):
    for i in range(len(xr)):
        r=sqrt((xr[i])**2+(yr[i])**2)
        theta = arctan(yr[i]/xr[i])
        vtheta=10
        if r < maxx:
            vx[i] = vtheta*sin(theta)*r + vx[i]
            vy[i] = vtheta*cos(theta)*r + vy[i]
            print vx[i],vy[i]
    return vx,vy

#initial setup

(neighbs,neighb_loc) = find_neighbors(xr,yr,h,n)#obtain init neighbors
rho = get_density(xr,yr,h,n,m,neighbs,neighb_loc)

#timestep stuff
t=0
dt=.01
maxt=5

j=0
saveImg=1

# time step stuff:
while t < maxt:
    
    #initialize vars:
    av = get_visc(xr,yr,h,vx,vy,rho,cs,neighbs,neighb_loc) #artificial viscosity
    
    xaccels, yaccels = get_accel(n,neighbs,neighb_loc,P,rho,m,xr,yr,h)
    xaccels = np.array([xaccels.flatten()]).T
    yaccels = np.array([yaccels.flatten()]).T
    xaccels, yaccels = boundf(xaccels,yaccels,xr,yr)

    #halfstep update
    vhx = vhx + dt*xaccels #v at t-(dt/2) to v at t+(dt/2)
    vhx = np.array([vhx.flatten()]).T
    vhy = vhy + dt*yaccels
    vhy = np.array([vhy.flatten()]).T
    #energies = get_energy(n,neighbs,neighb_loc,v,r,P,rho,av,h)
    #energies = bcs(energies,boundary,0.0)
 
    #vel update
    vx = vhx + 0.5*dt*xaccels
    vy = vhy + 0.5*dt*yaccels
    #vx,vy = f(xr,yr,vx,vy)
    
    #position update
    xr = xr + dt*vhx
    yr = yr + dt*vhy
    
    #neighbor update
    (neighbs,neighb_loc) = find_neighbors(xr,yr,h,n)
    
    #density update
    rho = get_density(xr,yr,h,n,m,neighbs,neighb_loc) 
    
    #pressure update
    P = k*rho
    
    #sound speed update
    #cs=sqrt(Gamma*B/rho)
    
    #time step update
    #dt = dts(csf,h,v,accels,cs)
   
    #output from timestep
    Output(j,P,xr,yr,t)
        
    t = t + dt
    j+=1
