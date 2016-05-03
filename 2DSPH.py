#!/usr/bin/env   b
#!/usr/bin/env python

#2D SPH code
#equations from Monaghan 2005

import sys
import numpy as np
import matplotlib.pyplot as mpl
import scipy.interpolate

###########################
####Variable Parameters####
###########################

n = 60 #sqrt num particles
maxx = 2 #radius
omega = .5 #rotational velocity
dt=.00001 #timestep; will decrease with increases in particle num, omega, kc. If you're getting a bug where
# the particles are all moving very far inward before rotating, decrease dt. A little bit of init inward movement is ok.
maxt=5 #total run time
kc = 600000 #spring constant of boundary

#Output file location
parentDir='/Users/sheacheatham/Research/2DSPH/run3'
imgDir='2DSPH'

minx = -maxx  
num = (maxx+abs(minx)) #number of grid boxes for neighbs search

######################
#### Functions #######
######################
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
  
    #this approach creates a fixed grid and only checks neighbor gridboxes for particles   
    boxes = loc_particles(xr,yr);
    top, bottom, left, right = edge_boxes()
    checkboxes = []
    xs=0
    ys=1
    for i in range(len(boxes)):
        if i in top:
            neighborBoxes=[i-1,i,i+1,i+(num-1),i+num,i+(num+1)]
        if i in bottom:
            neighborBoxes=[i-(num+1),i-num,i-(num-1),i-1,i,i+1]
        if i in left:
            neighborBoxes=[i-num,i-(num-1),i,i+1,i+num,i+(num+1)]
        if i in right:
            neighborBoxes=[i-1,i,i-num,i-(num+1),i+(num-1),i+num]
        else:
            neighborBoxes = [i-(num+1),i-num,i-(num-1),i-1,i,i+1,i+(num-1),i+num,i+(num+1)]
        for j in range(len(boxes[i])): #j is particle in question
            for k in range(len(boxes)):
                if k in neighborBoxes: #if one of the other boxes is a neighbor box
                    checkboxes.append(k) #just a check to make sure the right boxes are identified
                    for l in range(len(boxes[k])): #num particles in the neighbor box
                        dist = sqrt((boxes[i][j][xs]-boxes[k][l][xs])**2+(boxes[i][j][ys]-boxes[k][l][ys])**2)
                        if dist < 2.0*h:
                            neighb_loc[boxes[i][j][2],neighbs[boxes[i][j][2]]] = boxes[k][l][2]
                            neighbs[boxes[i][j][2]]+=1 #i is the box, j is the particle, 2 gives its
                            #floating'number' (the permanent number assigned to each particle)                   
                          
    return(neighbs,neighb_loc)        

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

#makes grid, determine which box each particle is in
def loc_particles(xr,yr):
    boxes = [ [] for n in range(num*num) ]
    for i in range(len(xr)):
        xkey = int(floor(xr[i]))
        ykey = int(floor(yr[i]))
        if xkey >= maxx:
            xkey = maxx-1
        if xkey <= minx:
            xkey = minx+1
        if ykey >= maxx:
            ykey = maxx-1
        if ykey <= minx:
            ykey = minx+1
        box = xkey+ykey+1+(2*maxx + 2)*((maxx-1) - ykey) #gives box number, given xkey and ykey
        position=xr[i],yr[i],i
        boxes[box].append(position)
    return boxes

#get smoothed density
def get_density(xr,yr,h,n,m,neighbs,neighb_loc):
    
    avrho=np.zeros(len(xr)) 
    for i in range(len(xr)):
        #for j in range(n):
        for j in range(neighbs[i]):
            rij = sqrt((xr[neighb_loc[i,j]]-xr[i])**2+(yr[neighb_loc[i,j]]-yr[i])**2)
            avrho[i]+= m[neighb_loc[i,j]]*kernel(rij,h)
    return avrho

#get artificial viscosity for each paricle i w/
#respect to each of its neighbors j
def get_visc(xr,yr,h,vx,vy,rho,cs,neighbs,neighb_loc):
    av = np.zeros((n*n,n*n))  
    #viscous constants
    alpha = 1.0
    beta = 2.0    
    for i in range(n*n):
        for j in range(neighbs[i]):
            vij = sqrt((vx[i]-vx[neighb_loc[i,j]])**2 + (vy[i]-vy[neighb_loc[i,j]])**2) 
            rij = sqrt((xr[neighb_loc[i,j]]-xr[i])**2+(yr[neighb_loc[i,j]]-yr[i])**2)
            if (vij*rij < 0.):
                mew = h * vij*rij/(rij**2 + (0.1*h)**2)
                cij = 0.5*(cs[i]+cs[neighb_loc[i,j]]) 
                rhoij = 0.5*(rho[i]+rho[neighb_loc[i,j]])
                av[i,neighb_loc[i,j]] = (-alpha*cij*mew + beta*mew**2)/rhoij #divide by zero eror?
    return av

#get acceleration for each particle
def get_accel(n,neighbs,neighb_loc,press,rho,m,xr,yr,h):   
    xaccels = np.zeros(len(xr))
    yaccels= np.zeros(len(xr))  
    for i in range(len(xr)):
        for j in range(neighbs[i]):
            rijx = xr[neighb_loc[i,j]]-xr[i]
            rijy = yr[neighb_loc[i,j]]-yr[i]
            k = neighb_loc[i,j]
            kernx, kerny = gradkernel(rijx,rijy,h)
            xaccels[i] -= m[k] * (press[i]/rho[i]**2 + press[k]/rho[k]**2 + av[i,k]) * kernx
            yaccels[i] -= m[k] * (press[i]/rho[i]**2 + press[k]/rho[k]**2 + av[i,k]) * kerny
    return xaccels, yaccels

#saving images
def Output(j,rho,xr,yr,t):
#    xrs  = np.reshape(xr,(n,n))
#    yrs  = np.reshape(yr,(n,n))
#    Ps = np.reshape(P,(n,n))
    mpl.figure(1)
    mpl.clf()
    #mpl.contour(xrs,yrs,Ps,offset=-100, cmap=cm.coolwarm)
    #circle=mpl.Circle((0,0),maxx,color='b',fill=False)#if you want to see where the boundary is
    mpl.plot(xr,yr,".")
    #mpl.gcf().gca().add_artist(circle)
    mpl.axis('equal')
    mpl.axis([minx-1,maxx+1,minx-1,maxx+1])
    #mpl.axis('equal')
    pause(0.0001)
    if j==1 or j % 2 == 0:
        jstr=str(j);
        while len(jstr)<4:
            jstr='0'+jstr
        imgfile=parentDir+imgDir+'img'+jstr+'.png'
        savefig(imgfile)

def POutput(xr,yr,P):
    #set up interpolation points
    xi, yi = np.linspace(minx, maxx, 100), np.linspace(minx, maxx, 100)
    xi, yi = np.meshgrid(xi, yi)   
    #Interpolate
    rbf = scipy.interpolate.Rbf(xr, yr, P, function='linear')
    Pi = rbf(xi, yi)
    mpl.figure(2)
    mpl.clf()
    mpl.imshow(Pi, vmin=0, vmax=100, origin='lower',extent=[minx-1, maxx+1, minx-1, maxx+1])
    mpl.axis('equal')
    mpl.colorbar()
    mpl.show()
    pause(0.0001)
    if j==1 or j % 1 == 0:
        jstr=str(j);
        while len(jstr)<4:
            jstr='0'+jstr
        imgfile=parentDir+'/press/'+imgDir+'Pressimg'+jstr+'.png'
        savefig(imgfile)
        
#cylindrical boundary
def cylboundf(xaccels,yaccels,xr,yr):
    for i in range(len(xr)):
        if xr[i]**2 > maxx**2-yr[i]**2:
            xaccels[i]= xaccels[i]-kc*xr[i]
        if yr[i]**2 > maxx**2-xr[i]**2:
            yaccels[i] = yaccels[i]-kc*yr[i]
    return xaccels, yaccels

def cylboundpos(xr,yr): #delete particles outside of the circle at t = 0
    indexx = []
    indexy = []
    i=len(xr)-1
    while i >= 0:
        if xr[i]**2+yr[i]**2 > maxx**2:
            indexx.append(xr[i])
            indexy.append(yr[i])            
            xr = np.delete(xr, i)
            yr = np.delete(yr, i)
        i-=1
    return xr,yr
         
def f(xr,yr,vx,vy): #rotational forcing function
    for i in range(len(vx)):
        r=sqrt(xr[i]**2+yr[i]**2)
        theta = arctan(yr[i]/xr[i])
        if r < maxx:
            if xr[i] >= 0:
                vx[i] -= omega*sin(theta)*r
                vy[i] += omega*cos(theta)*r
            elif xr[i]< 0:
                vx[i] += omega*sin(theta)*r
                vy[i] -= omega*cos(theta)*r
    return vx,vy

###############################
########Initial setup##########
###############################

xr= np.linspace(minx,maxx,n) 
yr = np.linspace(minx,maxx,n)
xr, yr = np.meshgrid(xr,yr)
xr, yr = np.array([xr.flatten()]).T, np.array([yr.flatten()]).T
xr,yr = cylboundpos(xr,yr)
xr, yr = np.array([xr.flatten()]).T, np.array([yr.flatten()]).T

vx = np.zeros(len(xr))
vy = np.zeros(len(xr)) 
vx = np.array([vx.flatten()]).T
vy = np.array([vy.flatten()]).T

vhx = np.zeros(len(xr)) #vs at half time step
vhy = np.zeros(len(xr))
vhx = np.array([vhx.flatten()]).T
vhy = np.array([vhy.flatten()]).T

#init analytical densities
rho = np.ones(len(xr))#(n*n)
rho = np.array([rho.flatten()]).T

dx = xr[1]-xr[0] #init space between two particles
dx = dx[0]
h = dx*2.0 #smoothing length; decrease with fewer particles
m = dx*rho[:] #mass per particle from density

#init pressure
k=1.0
P = k*rho

#speed of sound
cs=1/rho

(neighbs,neighb_loc) = find_neighbors(xr,yr,h,n)#obtain init neighbors
rho = get_density(xr,yr,h,n,m,neighbs,neighb_loc)

#timestep stuff
t=0
j=0
saveImg=1

#Add some noise to address particle condensing bug
noise = .01*np.random.normal(0,1,len(xr))
noise = np.array([noise.flatten()]).T

#### Algorithm #####

while t < maxt:
    
    #initialize vars:
    av = get_visc(xr,yr,h,vx,vy,rho,cs,neighbs,neighb_loc)
    
    xaccels, yaccels = get_accel(n,neighbs,neighb_loc,P,rho,m,xr,yr,h)
    xaccels = np.array([xaccels.flatten()]).T
    yaccels = np.array([yaccels.flatten()]).T
    xaccels, yaccels = cylboundf(xaccels,yaccels,xr,yr)

    #halfstep update
    vhx = vhx + dt*xaccels #v at t-(dt/2) to v at t+(dt/2)
    vhx = np.array([vhx.flatten()]).T
    vhy = vhy+ dt*yaccels
    vhy = np.array([vhy.flatten()]).T
    vhx,vhy = f(xr,yr,vhx,vhy)

    #vel update
    vx = vhx + 0.5*dt*xaccels
    vy = vhy + 0.5*dt*yaccels
    
    #position update
    xr = xr + dt*vhx
    yr = yr + dt*vhy
    
    #neighbor update
    (neighbs,neighb_loc) = find_neighbors(xr,yr,h,n)
    
    #density update
    rho = get_density(xr,yr,h,n,m,neighbs,neighb_loc)
    
    #pressure update
    P = k*rho
    
    #output
    Output(j,P,xr,yr,t)
    POutput(xr,yr,P)
    print(j,t,dt)
    
    t = t + dt
    j+=1
