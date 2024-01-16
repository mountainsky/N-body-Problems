import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as constant
from scipy.integrate import odeint
import astropy.units as u
import matplotlib.animation as animation
import scipy as sp 

#Defining constant values 
G = constant.G.value
Msun = constant.M_sun.value
Mearth= constant.M_earth.value
Mmoon= 7.34e24 #kg
Rsun = constant.R_sun.value
AU=constant.au.value


def euler_step(derivs, x, t, m, h, soft):
    """A single ODE step from t to t+h using the Euler method
    
    Args:
        derivs - a function that calculates the gravitational acceleration
        x - an array containing current values 
        t - current time at which derivs should be evaluated
        h - timestep
    """
    d = derivs(x, t, m, soft)
    x += d * h
    return x


def rk2_step(derivs, x, t, m, h, soft):
    """A single ODE step from t to t+h using the 2nd order Runge-Kutta method
    
    Args:
        derivs - a function that calculates the gravitational acceleration
        x - an array containing current values 
        t - current time at which derivs should be evaluated
        h - timestep
    """

    k1 = h * derivs(x, t, m, soft)
    k2 = h * derivs(x + 0.5 * k1, t + 0.5 * h, m, soft)
    x += k2

    return x


def rk4_step(derivs, x, t, m, h, soft):
    """A single ODE step from t to t+h using the 4th order Runge-Kutta method
    
    Args:
        derivs - a function that calculates the gravitational acceleration
        x - an array containing current values 
        t - current time at which derivs should be evaluated
        h - timestep
    """
    k1 = h * derivs(x, t, m, soft)
    k2 = h * derivs(x + 0.5 * k1, t + 0.5 * h, m, soft)
    k3 = h * derivs(x + 0.5 * k2, t + 0.5 * h, m, soft)
    k4 = h * derivs(x + k3, t + h, m, soft)
    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x

#Leapfrog method   
def leapfrog(derivs,x,t,m,h,soft):
    '''
    Solving a single step ODE by conserving energy using the Leapfrog method
    '''
    #Leapfrog from Euler
    #Euler half-step

    #x=x+0.5*h*derivs(x,t,m, soft)        
    #x_1 = x + h*derivs(x,t+0.5*h,m, soft)
    #x = x+ h*derivs(x_1,t+h,m, soft)  
    
    x_1 = x + 0.5*h*derivs(x,t+0.5*h,m, soft)     #Euler half-step
    x1 = x + h*derivs(x_1,t+h,m, soft)            #Leapfrog from Euler
    return x1



def integrator(derivs, p0, tarr, m, soft, stepper=euler_step, wrap=None):
    """ General function to integrate any ODE using step-by-step method

    Designed to have the same arguments as odeint
    
    Args:
        derivs - a function that calculates the gravitational acceleration
        p0 - an array containing initial phase space coords
        tarr - array of times at which n should be evaluated
        stepper - function used to advance to next timestep
        m - array of masses in order
        soft - softening parameter

    Returns:
        4D array with the coordinate vector corresponding to each timestep
    """

    # Create an empty array that we will fill as we go along
    dimensions = [len(tarr)] + list(p0.shape)
    parr = np.empty(dimensions)

    # Initialize to starting values of time and amount vector n
    t0 = tarr[0]

    p = p0
    parr[0] = p

    # Loop over all times
    for i in range(1, len(tarr)):
        t = tarr[i]
        h = t - t0

        # take a single step of size h using stepper
        p = stepper(derivs, p, t, m, h,soft)
        # for periodic boundary conditions, wrap coordinates
        if wrap is not None: p[:, 0, :] = np.remainder(p[:, 0, :] + wrap, wrap)
        parr[i] = p + 0
        t0 = t
    return parr

def Gravity(p_in, t, m, soft=0.):
    """
    Calculates phase space derivates (velocities and accelerations)
    
    Args:
        p : an array containing the phase space coords
        t : time
        m : an array of length p.shape[0] which gives the mass values of each shape in order
        soft: gravitational softening factor

    Returns:
        matched array with derivatives
        
    Note that phase space coords are assumed to be in order
    such that the first index gives the particle number, 
    the middle index whether it is position or velocity 
    and the final the 3 cartesian components
    """
    p = p_in
    
    #This if statement makes it compatible with odeint
    if len(np.shape(p_in)) == 1:
        n = len(p_in)//6
        p = np.empty((n,2,3))
        
        for i in range(n):
            p[i,0,:] = [p_in[(i*6)],p_in[(i*6+1)],p_in[(i*6+2)]]
            p[i,1,:] = [p_in[(i*6+3)],p_in[(i*6+4)],p_in[(i*6+5)]]

        bodies = p.shape[0]
        dpdt = np.zeros_like(p)
        for i in range(bodies):
            xi = p[i, 0, :]  # isolate xyz for particle i
            a = np.zeros(3)  # initialize acceleration for that particle

            # now loop over all other particles
            for j in range(bodies):
                xj = p[j, 0, :]  # isolate xyz for particle j
                if (i != j):
                    M = m[j]
                    if M == 0:
                    #Don't bother calculating acceleration from infinitesimal mass
                      continue
                    rvec = xj - xi  # separation vector
                
                    #Get dot product (norm squared) of separation
                    r2 = (np.sum(rvec*rvec)+soft*soft)
                    
                    #We want to keep directional part of r, so we raise r in to extra power in
                    # denominator, giving it a power of |r|^3 = r2^(3/2)=r2^(1.5).
                    # Just did it this way for efficiency sake
                    a += G*M*rvec/(r2**(1.5))

            dpdt[i, 1, :] = a  # derivative of velocity is a
            dpdt[i, 0, :] = p[i, 1, :]  # derivative position is v
            
        return dpdt.flatten()
        

    bodies = p.shape[0]
    dpdt = np.zeros_like(p)
    for i in range(bodies):
        xi = p[i, 0, :]  # isolate xyz for particle i
        a = np.zeros(3)  # initialize acceleration for that particle

        # now loop over all other particles
        for j in range(bodies):
            xj = p[j, 0, :]  # isolate xyz for particle j
            if (i != j):
                M = m[j]
                if M == 0:
                    #Don't bother calculating acceleration from infinitesimal mass
                    continue
                rvec = xj - xi  # separation vector
            
                #Get dot product (norm squared) of separation
                r2 = (np.sum(rvec*rvec)+soft*soft)
                
                #We want to keep directional part of r, so we raise r in to extra power in
                # denominator, giving it a power of |r|^3 = r2^(3/2)=r2^(1.5).
                # Just did it this way for efficiency sake
                a += G*M*rvec/(r2**(1.5))

        dpdt[i, 1, :] = a  # derivative of velocity is a
        dpdt[i, 0, :] = p[i, 1, :]  # derivative position is v

    return dpdt

class NBody(object):
    def __init__(self, p0, m,  dpdt=Gravity, soft=0., wrap=None):
        """
        Args:
            p0: Initial phase space array
            m: Array of object masses that match order of p0
            soft: Gravitational softening constant
            dpdt: Function to calculate phase space derivatives
            wrap: boundary condition modifier
        """
        self.p0 = p0
        self.dpdt = dpdt
        self.wrap = wrap
        self.m = m
        self.soft = soft

    def integrate(self, times, stepper=rk4_step):
        return integrator(self.dpdt,
                          self.p0,
                          times,
                          self.m,
                          self.soft,
                          stepper=stepper,
                          wrap=self.wrap)


def Earth_and_sun(x1,y1,vx1,vy1,times, method='euler', wrap=None):

    pos_E = [x1, y1, 0.]
    vel_E = [vx1, vy1, 0.]
    
    pos_sun = [0., 0., 0.]
    vel_sun = [0., 0., 0.]

    n = 2 
    # make one big phase space array
    p0 = np.empty((n,2,3))
    p0[0,0,:] = pos_E
    p0[0,1,:] = vel_E
    
    p0[1,0,:] = pos_sun
    p0[1,1,:] = vel_sun
    
    steppers = {'euler': euler_step,'RK2': rk2_step,'RK4': rk4_step, 'leapfrog':leapfrog}
    stepper = steppers[method]
    m=np.array([Mearth,Msun])
    soft = 0.1
    
    project = NBody(p0, m, soft=0., dpdt=Gravity, wrap=wrap)
    
    return project.integrate(times, stepper)

def Earth_and_sun_plots(x1, y1, vx1, vy1, timestep = 86000, method = 'RK4'):

    years = 1.0
    totaltime = 3.154e7*years
    times = np.arange(0., totaltime, timestep)

    title = f'h = {timestep:5.2f};  method = {method}'
    print(title)
    
    out = Earth_and_sun(x1, y1, vx1, vy1, times, method=method)
    x0 = out[:,0,0,0]
    y0 = out[:,0,0,1]
    vx0 =  out[:,0,1,0]
    vy0 =  out[:,0,1,1]
    x1 = out[:,1,0,0]
    y1 = out[:,1,0,1]
    vx1 =  out[:,1,1,0]
    vy1 =  out[:,1,1,1]

    plt.plot(x0, y0, ls='-', ms=3, label='Earth')
    plt.plot(x1, y1, 'o', ls='-', ms=10, label='Sun')
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')
    plt.xlim(-2.1e11,2.1e11)
    plt.ylim(-2.1e11,2.1e11)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.savefig('349 Euler method.png', bbox_inches='tight',pad_inches=0.2,dpi=1000)
    plt.show()

#for method in ['euler', 'RK2', 'RK4','leapfrog']:
    #Earth_and_sun_plots(1.5e11,0.,0.,30000,method=method)



#Validation test case 
def sun_odeint():
    '''
    Solving the ODE using the built-in odeint function from scipy.integrate
    Using odeint function to test the acurracy of results
    '''
    x1,y1,vx1,vy1 = AU,0.,0.,29784.8
    soft=0
    tyr=1  
    t=np.arange(0,3.154e7*tyr,1e5)
    #Array of Sun and Earth masses 
    m=np.array([Mearth,Msun])
    
    p0 = np.array([x1,y1,0,vx1,vy1,0,0,0,0,0,0,0])
    #Let p0 be formatted as [x1,y1,z1,vx1,vy1.vz1,x2,y2,z2,vx2,vy2,vz2,etc.]
    #odeint returns flattened array
    pf = odeint(Gravity,p0,t,tfirst=False,args=(m,soft))
    
    #Get flattened phase array into multidimensional array
    n = np.shape(pf)[1]//6
    steps = np.shape(pf)[0]
    parr = np.empty([steps,n,2,3])
    
    #Loop over timesteps
    for step in range(steps):
        p = np.empty([n,2,3])
        for i in range(n):
            p[i,0,:] = [pf[step][i*6],pf[step][i*6+1],pf[step][i*6+2]]
            p[i,1,:] = [pf[step][i*6+3],pf[step][i*6+4],pf[step][i*6+5]]
        parr[step] = p
    
    x0 = parr[:,0,0,0]
    y0 = parr[:,0,0,1]
    x1 = parr[:,1,0,0]
    y1 = parr[:,1,0,1]
    
    #Plot odeint
    plt.plot(x0, y0, ls='-', ms=3, label='Earth')
    plt.plot(x1, y1, 'o', ls='-', ms=10, label='Sun')
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')
    plt.xlim(-2.1e11,2.1e11)
    plt.ylim(-2.1e11,2.1e11)
    plt.legend(loc='upper left')
    plt.title('Odeint method')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


#"Studies Section"

class Galaxy():
    def __init__(self, M, N1, N2, N3, x, y, z, vx, vy, vz, name = "", spin =1):
        """
        Initializes a galaxy with stars in circular orbit around the center
        
        Parameters:
        -----------
        M = mass in kg
        N1 = total stars at r=5kpc
        N2 = total stars at r=10kpc
        N3 = total stars at r=15kpc
        x,y,z = 3D coordinates for center of galaxy in kpc
        vx,vy,vz = velocity of entire galaxy in kpc/million years
        spin = -1 or 1. 1 for clockwise spinning galaxy, -1 for counterclockwise
        """
        #First convert units
        x = (x*u.kpc).to('m')
        y = (y*u.kpc).to('m')
        z = (z*u.kpc).to('m')
        vx = (vx*u.kpc/(1e6*u.yr)).to('m/s')
        vy = (vy*u.kpc/(1e6*u.yr)).to('m/s')
        vz = (vz*u.kpc/(1e6*u.yr)).to('m/s')
        orbit1 = (5*u.kpc).to('m')
        orbit2 = (10*u.kpc).to('m')
        orbit3 = (15*u.kpc).to('m')
        
        #Initialize
        self.x = x.value
        self.y = y.value
        self.z = z.value
        self.vx = vx.value
        self.vy = vy.value
        self.vz = vz.value
        self.name = name
        self.mass = M
        stars = []
        scount = [N1,N2,N3]
        orbits = [orbit1.value,orbit2.value,orbit3.value]
        for i in range(len(scount)):
            r = orbits[i]
            N = scount[i]
            sx = np.random.uniform(-r,r,N)
            sy = np.sqrt(r*r-sx*sx) * np.random.choice([-1,1],size=len(sx))
            
            v = np.sqrt(G*self.mass/r) #Get the speed of a circular orbit
            print(f"Orbital velocity of {v:.3f} m/s for {self.name} Galaxy at r={(r*u.m).to('kpc').value:.3f}kpc")
            phi=np.arctan(sy/sx)
            svx = v*np.sin(phi)
            svy = v*np.cos(phi)
            
            #Rotate stars all in same direction - will start clockwise. Change sign of all terms to go counter-clockwise
            for j in range(len(svx)):
                if sx[j] <= 0 and sy[j] < 0:
                    #Bottom left quadrant
                    svx[j] = np.abs(svx[j])*-spin
                    svy[j] = np.abs(svy[j])*spin
                if sx[j] >= 0 and sy[j] > 0:
                    #Top right quadrant
                    svx[j] = np.abs(svx[j])*spin
                    svy[j] = np.abs(svy[j])*-spin
                if sx[j] < 0 and sy[j] >= 0:
                    #Top left quadrant
                    svx[j] = np.abs(svx[j])*spin
                    svy[j] = np.abs(svy[j])*spin
                if sx[j] > 0 and sy[j] <= 0:
                    #Bottom right quadrant
                    svx[j] = np.abs(svx[j])*-spin
                    svy[j] = np.abs(svy[j])*-spin
            for s in range(N):
                p = np.zeros([2,3])
                #Set position of phase vector
                p[0,:] = [sx[s]+self.x,sy[s]+self.y,0]
                #Set velocities of phase vector
                p[1,:] = [svx[s]+self.vx,svy[s]+self.vy,0]
                stars.append(p)
        self.stars = np.array(stars)
    
    def plot_gal(self):
        """
        Plots the galaxy in blue and the stars in yellow

        """
        plt.clf()
        starx = []
        stary = []
        for star in self.stars:
            starx.append(star[0][0])
            stary.append(star[0][1])
            
        plt.scatter((starx*u.m).to('kpc').value,(stary*u.m).to('kpc').value,s=10,color='y',label="stars")
        plt.scatter((self.x*u.m).to('kpc').value,(self.y*u.m).to('kpc').value,s=30,color='b',label="galaxy")
        plt.title(f"{self.name} galaxy")
        plt.xlabel('$x$ (kpc)')
        plt.ylabel('$y$ (kpc)')
        plt.legend()
        plt.show()

def plot_galaxies(g1,g2,name="Galaxies"):
    plt.clf()
    starx1 = []
    stary1 = []
    starx2 = []
    stary2 = []
    for star in g1.stars:
        starx1.append(star[0][0])
        stary1.append(star[0][1])
    for star in g2.stars:
        starx2.append(star[0][0])
        stary2.append(star[0][1])
    plt.xlabel('$x$ (kpc)')
    plt.ylabel('$y$ (kpc)')
    plt.scatter((starx1*u.m).to('kpc').value,(stary1*u.m).to('kpc').value,s=10,color='y')
    plt.scatter((g1.x*u.m).to('kpc').value,(g1.y*u.m).to('kpc').value,s=30,color='b')
    plt.scatter((starx2*u.m).to('kpc').value,(stary2*u.m).to('kpc').value,s=10,color='y')
    plt.scatter((g2.x*u.m).to('kpc').value,(g2.y*u.m).to('kpc').value,s=30,color='b')
    plt.title(f"{name}")
    plt.show()
    
def Gcollide(Gmain,Gpert,times,soft=0.1,method='euler'):
    #Galaxy collision
    #Will Assume they are in the same plane
    
    #Convert time array from million years into seconds
    times = (times*1e6*u.yr).to('s').value
    
    #Perturber Galaxy
    pos_P = [Gpert.x, Gpert.y, Gpert.z]
    vel_P = [Gpert.vx, Gpert.vy, Gpert.vz]
    
    #Main Galaxy at origin
    pos_G = [Gmain.x, Gmain.y, Gmain.z]
    vel_G = [Gmain.vx, Gmain.vy, Gmain.vz]

    NG = len(Gmain.stars)
    NP = len(Gpert.stars)
    
    pGal = np.empty((2,2,3))
    pGal[0,0,:] = pos_G
    pGal[0,1,:] = vel_G
    pGal[1,0,:] = pos_P
    pGal[1,1,:] = vel_P
    
    # make one big phase space array with all stars
    p0 = np.array(list(pGal)+list(Gmain.stars)+list(Gpert.stars))
    steppers = {'euler': euler_step,'RK2': rk2_step,'RK4': rk4_step, 'leapfrog':leapfrog}
    stepper = steppers[method]
    
    smass = list(np.zeros(NG+NP))
    m=np.array([Gmain.mass,Gpert.mass]+smass)
    
    collision = NBody(p0, m, soft=soft, dpdt=Gravity, wrap=None)
    return collision.integrate(times, stepper)


def Gcollision_plots(Gmain,Gpert, ttotal = 800, timestep = 1,soft=0.1, method = 'leapfrog'):
    """
    NOT USED, replaced by AnimatedGcollide
    ttotal and timestep give in units of a million years
    """
    times = np.arange(0., ttotal, timestep)
    title = f'h = {timestep:5.2f};  method = {method}'
    print(title)
    out = Gcollide(Gmain,Gpert,times, soft=soft,method=method)
    for n in range(np.shape(out)[1]):
        xn = (out[:,n,0,0]*u.m).to('kpc').value
        yn = (out[:,n,0,1]*u.m).to('kpc').value
        if n == 0 or n ==1:
            plt.plot(xn, yn, marker='o', ls='-', linewidth=0.5, ms=2, label="Galaxy "+str(n))
        else:
            plt.plot(xn, yn,  color='y',ls='-', ms=1, linewidth=0.2)
            
    plt.xlabel('$x$ (kpc)')
    plt.ylabel('$y$ (kpc)')
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    plt.legend()
    plt.title(title)
    #plt.savefig('galaxycollision.png', bbox_inches='tight',pad_inches=0.2,dpi=1000)
    plt.show()

class AnimatedGcollide(object):
    def __init__(self,Gmain,Gpert,ttotal=1500,timestep=3,soft=3e12,method='leapfrog',testout=None,bounds=[130,100],aniname="Galaxy_Collision",info=1):
        """
        ttotal and timestep give in units of a million years
        testout is just using previously saved outputs to test this class quicker
        Bounds is the plotting limits in kpc [xsize,ysize]
        """
        self.Gmain = Gmain
        self.Gpert = Gpert
        self.info = info
        times = np.arange(0., ttotal, timestep)
        self.times=times
        
        #Get output or use already generated output:
        self.out = testout
        if testout is None:
            self.out=Gcollide(Gmain,Gpert,times, soft=soft,method=method)
        
        self.l = len(self.out[:,0,0,0]) # number of frames (parameter for the animate function)
        
        fig, ax = plt.subplots()
        ax.set_facecolor('xkcd:black')
        if info ==1:
            self.textR = ax.text(-125, 82,"",color='white',fontsize=8)
            self.textT = ax.text(-125, 90,"",color='white',fontsize=8)
        self.fig = fig
        self.ax = ax
        self.ax.set_xlim(-bounds[0],bounds[0])
        self.ax.set_ylim(-bounds[1],bounds[1])
        plt.xlabel('$x$ (kpc)')
        plt.ylabel('$y$ (kpc)')
        plt.title(f'{aniname}')
        
        self.indexing = len(Gmain.stars) #Format indexing
        self.scatter = [] #Reset animation data
        self.aniname = aniname #Animation title
        self.ani = animation.FuncAnimation(self.fig, 
                                           self.update, 
                                           frames=range(self.l), 
                                           interval=30, 
                                           init_func=self.setup_plot, 
                                           blit=True)
        
    def setup_plot(self):
        names = [self.Gmain.name,self.Gpert.name]
        colors = ['blue','red']
        self.scatter = []
        for n in range(np.shape(self.out)[1]):
            if n < 2:
                line, = self.ax.plot([], [], marker='o', ms=6, label=names[n]+" Galaxy", color=colors[n])
                self.scatter.append(line)
            elif n >= 2 and n < self.indexing+2:
                line, = self.ax.plot([], [], marker='o', color='y', ms=2)
                self.scatter.append(line)
            else:
                line, = self.ax.plot([], [], marker='o', color='chartreuse', ms=2)
                self.scatter.append(line)
        return self.scatter
        
    def update(self, i):
        for n in range(np.shape(self.out)[1]):
            xn_i = (self.out[:,n,0,0][i]*u.m).to('kpc').value
            yn_i = (self.out[:,n,0,1][i]*u.m).to('kpc').value
            self.scatter[n].set_data(xn_i,yn_i)
        if self.info==1:
            x1 = (self.out[:,0,0,0][i]*u.m).to('kpc').value
            x2 = (self.out[:,1,0,0][i]*u.m).to('kpc').value
            y1 = (self.out[:,0,0,1][i]*u.m).to('kpc').value
            y2 = (self.out[:,0,0,1][i]*u.m).to('kpc').value
            self.textR.set_text(f"r={np.sqrt((x2-x1)**2+(y2-y1)**2):.1f}kpc") 
            self.textT.set_text(f"Time Elapsed:{self.times[i]:.1f}e6yr")
        return self.scatter
    
    def save_anim(self):
        with open(f"{self.aniname}.html", "w") as f:
            print(self.ani.to_html5_video(), file=f)
            
def escV(M1,M2,r):
    '''
    M1,M2 given in kg
    r given in kpc
    Calculate speed needed for parabolic trajectory based on mass and radial distance
    '''
    r = (r*u.kpc).to('m').value
    v = np.sqrt(2*G*(M1+M2)/r)
    v = (v*u.m/u.s).to('kpc/yr').value * 1e6
    return v


def generate_gcanims(radii,stars=90,ttotal=1400,timestep=1):
    '''
    Run various animation tests for different impact parameters
    radii = array or list of y-distance between galactic centers given in kpc
    stars = total number of stars evenly distributed between 3 orbits
    '''
    M1 = Msun*10**11
    M2 = Msun*10**10
    s = stars//3
    #Galactic softening parameter
    Gsoft = (0.1*u.kpc).to('m').value     
    
    for r in radii:
        #Let the total distance be 100kpc
        x = np.sqrt(100*100 - r*r)
        Vesc = escV(M1,M2,10)
        Gmain = Galaxy(M1,s,s,s+stars%3,0,0,0,0,0,0,name="Main",spin=1)
        GmainCCW = Galaxy(M1,s,s,s+stars%3,0,0,0,0,0,0,name="Main",spin=-1)
        Gpert = Galaxy(M2,s,s,s+stars%3,x,r,0,-Vesc,0,0,name="Perturber",spin=1)  
        GpertCCW = Galaxy(M2,s,s,s+stars%3,x,r,0,-Vesc,0,0,name="Perturber",spin=-1) 
        
        plot_galaxies(Gmain,Gpert,name=f"r_b={r}kpc, {stars} stars each")
        #Test 1-1: spin all same direction 
        #Generate animation
        gcanim = AnimatedGcollide(Gmain,Gpert,method='leapfrog',ttotal=ttotal,timestep=timestep,soft=Gsoft,aniname=f"r_b={r}kpc,spin-cw")
        gcanim.save_anim()
        plt.show()
        
        #Test 1-2 spin all different direction
        #Generate animation
        gcanim = AnimatedGcollide(GmainCCW,GpertCCW,method='leapfrog',ttotal=ttotal,timestep=timestep,soft=Gsoft,aniname=f"r_b={r}kpc,spin-ccw")
        gcanim.save_anim()
        plt.show()        
        
        #Test 1-3 spin CW and CCW
        #Generate animation
        gcanim = AnimatedGcollide(Gmain,GpertCCW,method='leapfrog',ttotal=ttotal,timestep=timestep,soft=Gsoft,aniname=f"r_b={r}kpc,spin-cw vs. spin-ccw")
        gcanim.save_anim()
        plt.show()        
        
        #Test 1-4 spin CCW and CW
        #Generate animation
        gcanim = AnimatedGcollide(GmainCCW,Gpert,method='leapfrog',ttotal=ttotal,timestep=timestep,soft=Gsoft,aniname=f"r_b={r}kpc,spin-ccw vs. spin-cw")
        gcanim.save_anim()
        plt.show()
        
# =============================================================================
#testing code for single animation
# Gmass1 = Msun*10**11
# Gmass2 = Msun*10**10   
# Vesc = escV(Gmass1,Gmass2,np.sqrt(95*95+31.2245*31.2245))
# #Generate galaxies
# Gmain = Galaxy(Gmass1,20,25,30,0,0,0,0,0,0,name="Main",spin=1)
# Gpert = Galaxy(Gmass2,20,25,30,95,31.2245,0,-Vesc,0,0,name="Perturber",spin=1)  
# GmainCCW = Galaxy(Gmass1,20,25,30,0,0,0,0,0,0,name="Main",spin=-1)
# GpertCCW = Galaxy(Gmass2,20,25,30,95,31.2245,0,-Vesc,0,0,name="Perturber",spin=-1)  
# 
# Gmain.plot_gal()
# Gpert.plot_gal()
# plot_galaxies(Gmain,Gpert)
# Gnull = Galaxy(0,0,0,0,95,5,0,0,0,0,name="Null")
# 
# #Galactic softening parameter
# Gsoft = (0.1*u.kpc).to('m').value
# 
# #Generate animation
# ttotal = 1500
# timestep=3
# times = np.arange(0., ttotal, timestep)
# #Testout just used to to save output when working in console to make it more efficient for future runs
# testout=Gcollide(Gmain,Gpert,times,soft=Gsoft,method='RK4')
# gcanim = AnimatedGcollide(Gmain,Gpert,method='RK4', ttotal=ttotal,timestep=timestep,testout=testout, bounds=[130,100], info=1)
# gcanim.save_anim()
# plt.show()
# 
# =============================================================================

# =============================================================================
# #Generates video of main galaxy alone
# Gmain = Galaxy(Msun*10**11,20,25,30,0,0,0,0,0,0,name="Main",spin=1)
# Gnull = Galaxy(0,0,0,0,100,100,0,0,0,0)
# #Generate animation
# gcanim = AnimatedGcollide(Gmain,Gnull,method='RK4',bounds=[20,20],aniname="Main Galaxy")
# gcanim.save_anim()
# plt.show()
# =============================================================================
rtest = [15,30,45,60]
generate_gcanims(rtest)

# Commented out IPython magic to ensure Python compatibility.
### 3 body scattering ###

import matplotlib.animation as animation

# so that the animations work on Jupyter notebook
# %matplotlib notebook 

m1 = Msun
m2 = Msun
m3 = 2.0*Msun
M = m1*m2/(m1+m2)

r = 2*AU
v = np.sqrt(G*M/r)

years = 10.0
totaltime = 3.154e7*years
times = np.arange(0., totaltime, 86000*years)

def ThreeBodyScattering(times, xi = 15.0*AU, yi = 0.45*AU, vxi = -20000, method='RK4', wrap=None):
    
    '''
    Calculates the positions and velocities for a 3 body scattering system.
    A binary star system starts out in a stable orbit.
    A third star approaches with some velocity.
    The initial conditions of this third star can be varied.
    
    Parameters
    ----------------------------
    times - an array of the times
    xi - initial x position of the third star (default is 15.0 AU)
    yi - initial y position of the third star (default is 0.45 AU)
    vxi - initial x velocity of the third star (default is -20000 m/s)
    method - integration method (default is 'RK4')
    
    Example
    ----------------------------
    >>>years, totaltime = 10.0, 3.154e7*years
    >>>times = np.arange(0., totaltime, 86000*years)
    >>>out = ThreeBodyScattering(times, xi = 15.0*AU, yi = 0.45*AU, vxi = -20000, method='RK4')
    returns an array with the positions and velocities for each star corresponding to each time step in *times*.
    '''
    
    IC_pos = np.array([[-AU, 0., 0.], # position of star 1 in binary system
                       [AU, 0., 0.], # position of star 2 in binary system
                       [xi, yi, 0.]]) # position of third star, disturbing the binary system
    IC_vel = np.array([[0., -v, 0.], # position of star 1 in binary system
                       [0., v, 0.], # position of star 2 in binary system
                       [vxi, 0., 0.]]) # position of third star, disturbing the binary system
    
    p0 = np.empty((3,2,3)) 
    for j in range(3):
        p0[j,0,:] = IC_pos[j]
        p0[j,1,:] = IC_vel[j]
    
    steppers = {'euler': euler_step,'RK2': rk2_step,'RK4': rk4_step}
    stepper = steppers[method]
    
    m = np.array([m1,m2,m3])
    soft = 0.1
    
    project = NBody(p0, m, soft=0., dpdt=Gravity, wrap=wrap)
    return project.integrate(times, stepper)

class AnimatedScattering(object):
    
    '''
    Creates an animation of the three body scattering problem.
    
    Parameters
    ----------------------------
    method - integration method (default is 'RK4')
    xi - initial x position of the third star (default is 15.0 AU)
    yi - initial y position of the third star (default is 0.45 AU)
    vxi - initial x velocity of the third star (default is -20000 m/s)
    plotsize - the bounds of the x and y axes that are displayed in the animation (default is 10 AU)
    
    Returns
    ----------------------------
    The binary star system is animated as a small yellow and a medium orange star in a stable orbit.
    The third star is animated as a large blue star. It's initial conditions are displayed.
    The energies (kinetic, potential, and total) of the system are displayed for each frame/time step.
    
    Example
    ----------------------------
    >>>AnimatedScattering(xi = 15*AU, yi = 0.45*AU, vxi=-20000)
    >>>plt.show()
    '''
    
    def __init__(self, method='RK4', xi=15*AU, yi=0.0*AU, vxi=-20000, plotsize=10*AU):
        '''
        Obtains an array of positions and velocities of the stars using the ThreeBodyScattering() function.
        Initializes the figure and creates the animation.
        '''
        
        self.vxi = vxi
        self.xi = xi
        self.yi = yi
        
        # the actual data
        self.out = ThreeBodyScattering(times, xi=self.xi, yi=self.yi, vxi=self.vxi, method=method)
        
        self.l = len(self.out[:,0,0,0]) # number of frames (parameter for the animate function)
        
        fig, ax = plt.subplots(dpi=300)
        self.fig = fig
        self.ax = ax
        
        self.ax.set_title('3 Body Scattering')
        self.ax.set_xlim(-plotsize/AU, plotsize/AU)
        self.ax.set_ylim(-plotsize/AU, plotsize/AU)
        
        plt.xlabel('$x$ (AU)')
        plt.ylabel('$y$ (AU)')
        
        # set up text to displays the energies
        self.textKE = ax.text(-1.4e12/AU, -0.9e12/AU, '')
        self.textPE = ax.text(-1.4e12/AU, -1.1e12/AU, '')
        self.textE = ax.text(-1.4e12/AU, -1.3e12/AU, '')
        
        # animation
        self.ani = animation.FuncAnimation(self.fig, 
                                           self.update, 
                                           frames=range(self.l), 
                                           interval=50, #50
                                           init_func=self.setup_plot, 
                                           blit=True)
        
    def setup_plot(self):
        '''
        Sets up a plot for each star.
        '''
        
        # setting up plots
        # the third star gets labelled with its initial conditions
        self.line0, = plt.plot([], [],c='orange', marker='o', ms=10)
        self.line1, = plt.plot([], [],c='gold', marker='o', ms=7)
        self.line2, = plt.plot([], [],c='lightblue', marker='o', ms=15, 
                               label=f'Initial conditions: $x={self.xi/AU}$ $AU,y={self.yi/AU}$ $AU,v_x={self.vxi}$ $m/s$')
        self.ax.legend(loc='upper left')
        return self.line0, self.line1, self.line2,
        
    def update(self, i):
        
        '''
        Obtains the position of each star at each frame and plots it.
        Calculates the kinetic, potential, and total energy for the system at each frame and displays it.
        
        Parameters
        ----------------------------
        i - frame number
        '''
        
        # positions at the ith frame
        x1, y1 = self.out[:,0,0,0][i], self.out[:,0,0,1][i]
        x2, y2 = self.out[:,1,0,0][i], self.out[:,1,0,1][i]
        x3, y3 = self.out[:,2,0,0][i], self.out[:,2,0,1][i]
        
        # plotting the positions at the ith frame
        self.line0.set_data([x1/AU, y1/AU])
        self.line1.set_data([x2/AU, y2/AU])
        self.line2.set_data([x3/AU, y3/AU])
        
        # kinetic energy
        KE1 = 0.5*m1*((self.out[:,0,1,0][i])**2 + (self.out[:,0,1,1][i])**2)
        KE2 = 0.5*m1*((self.out[:,1,1,0][i])**2 + (self.out[:,1,1,1][i])**2)
        KE3 = 0.5*m1*((self.out[:,2,1,0][i])**2 + (self.out[:,2,1,1][i])**2)
        
        # distances between the stars
        r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        r31 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
        
        # potential energy
        PE12 = -G*m1*m2/r12
        PE23 = -G*m2*m3/r23
        PE31 = -G*m3*m1/r31
        
        KE = KE1 + KE2 + KE3
        PE = PE12 + PE23 + PE31
        E = KE + PE
        
        self.textKE.set_text(f'Kinetic energy = {KE:0.3E} J') # display the kinetic energy
        self.textPE.set_text(f'Potential energy = {PE:0.3E} J') # display the potential energy
        self.textE.set_text(f'Total energy = {E:0.3E} J') # display the total energy
        
        return self.line0, self.line1, self.line2,

    def save_anim(self):
        FFwriter = animation.FFMpegWriter(fps=24)
        self.ani.save('ThreeBodyScattering - NewBinarySystem.mp4', writer = FFwriter)

# xi = 15*AU, yi = 1.0*AU, vxi=-20000 weird initial conditions
scattering_animation = AnimatedScattering(xi = 15*AU, yi = 0.45*AU, vxi=-20000)
#scattering_animation.save_anim()
plt.show()

# Commented out IPython magic to ensure Python compatibility.
### Plotting trajectories of the stars - Not animated ###
# This is so we could add images to the report.

# %matplotlib inline
xi = 40*AU
yi = 0.0*AU
vxi=-100000
out = ThreeBodyScattering(times, xi = xi, yi = yi, vxi=vxi, method='leapfrog')

plt.plot(out[:,0,0,0]/AU, out[:,0,0,1]/AU,'o', ls='-', ms='4', c='orange', linewidth=1, markevery=5)
plt.plot(out[:,1,0,0]/AU, out[:,1,0,1]/AU,'o', ls='-', ms='4', c='gold', linewidth=1, markevery=5)
plt.plot(out[:,2,0,0]/AU, out[:,2,0,1]/AU,'o', ls='-', ms='4', c='lightblue',linewidth=1, markevery=5, 
         label=f'Initial conditions: $x={xi/AU}$ $AU,y={yi/AU}$ $AU,v_x={vxi}$ $m/s$')
plt.legend(loc='upper left')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title('3 Body Scattering')
plt.xlabel('$x$ $(AU)$')
plt.ylabel('$y$ $(AU)$')
#plt.savefig('ThreeBodyScattering - Fast.png', bbox_inches='tight',pad_inches=0.2, dpi=300)
plt.show()

# Lagrange Points within the Earth-Sun system 
# 3 body system: Sun-Earth-Moon (has zero mass)

m=np.array([Msun,Mearth,Mmoon])
u=m[0]/(m[0]+m[1])
u2= m[1]/(m[0]+m[1])

#Collinear points
def l1(x):
    return -x-(1-u)/(x+u)**2+u/(x-1+u)**2
def l2(x):
    return x-(1-u)/(x+u)**2-u/(x-1+u)**2
def l3(x):
    return x+(1-u)/(x+u)**2+u/(x-1+u)**2
L1=sp.optimize.newton(l1,1)
L2=sp.optimize.newton(l2,1)
L3=sp.optimize.newton(l3,1)
#Triangular points have analytical solutions
#L4=[1/2-u, np.sqrt(3)/2]
#L5=[1/2-u, -np.sqrt(3)/2]
L4=[1/2-u2, np.sqrt(3)/2]
L5=[1/2-u2, -np.sqrt(3)/2]


#Plot 
t=np.linspace(0,2*np.pi)
plt.plot(u*np.cos(t),u*np.sin(t),'-b',lw=0.5)
plt.axhline(0, color='k', ls='-',lw=0.5)
#Center of mass Earth-sun system
plt.plot(0,0,marker="+", ms=15, color="black")
plt.plot(u, 0,marker='o',color='blue', ms=8, label='Earth')
plt.plot(-u2, 0,'o',color='orange', ms=10, label='Sun')
plt.title('Lagrange points in the Earth-Sun system')
plt.plot(L1,0,'o', color="g",ms=5,label="L1")
plt.plot(L2, 0,'o',color='r', ms=5, label='L2')
plt.plot(L3, 0,'o',color='gray', ms=5, label='L3')
plt.plot(L4[0],L4[1],'o',color='c', ms=5, label='L4')
plt.plot(L5[0],L5[1],'o',color='m', ms=5, label='L5')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot([u, -0.5+u, 1-u, -0.5+u,u], [0, -np.sqrt(3)/2, 0,np.sqrt(3)/2, 0], 'k', ls="--", lw=1)
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Appendix B

#Lagrange Points Study case 
def eqmotion(p,t):
    rx,ry,rz,vx,vy,vz=p[0],p[1],p[2],p[3],p[4],p[5]
    #r1=[rx+u,ry,rz]
    r1=[rx+u,0,0]
    #r2=[rx-1+u,ry,rz]
    r2=[rx-1+u,0,0]
    
    #Updating
    parr=np.zeros(6)
    parr[:3]=p[3:]
    parr[3]=2*vy
    parr[4]=-2*vx
    parr[5]=0
    return parr.flatten()

def Lagrange_points(): 
    '''
    Determine Lagrange points by solving the ODE
   
    L2:Unstable Lagrange point is located along the Sun-Earth line (unstable appx. 23 days)
    L4:Stable point is one of the forming apex of the equilateral
       triangles with the sun and Earth in the base vertices (L4 leads Earth's orbit) 
    '''
    x1,y1,vx1,vy1 = AU,0.,0.,29784.8
    tyr=1  
    t=np.arange(0,3.154e7*tyr,1e5)
    p0 = np.array([x1,y1,0,vx1,vy1,0])
    #odeint returns flattened array
    sol = odeint(eqmotion,p0,t,tfirst=False)
    return sol

z=Lagrange_points()
#Plot ODE solutions
#Get flattened phase array into multidimensional array
n = np.shape(z)[1]//6
steps = np.shape(z)[0]
parr = np.empty([steps,n,2,2])

#Loop over timesteps
for step in range(steps):
    p = np.empty([n,2,2])
    for i in range(n):
        p[i,0,:] = [z[step][i*6],z[step][i*6+1],z[step][i*6+2]]
        p[i,1,:] = [z[step][i*6+3],z[step][i*6+4],z[step][i*6+5]]
    parr[step] = p