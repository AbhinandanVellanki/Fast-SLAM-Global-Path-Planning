from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy

#plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def initialize_particles(num_particles, num_landmarks):
    #initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        #initialize pose: at the beginning, robot is certain it is at [0,0,0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        #initial weight
        particle['weight'] = 1.0 / num_particles
        
        #particle history aka all visited poses
        particle['history'] = []

        #initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            #initialize the landmark mean and covariance 
            landmark['mu'] = [0,0]
            landmark['sigma'] = np.zeros([2,2])
            landmark['observed'] = False

            landmarks[i+1] = landmark

        #add landmarks to particle
        particle['landmarks'] = landmarks

        #add particle to set
        particles.append(particle)

    return particles

def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.4, 0.4, 0.2, 0.2]

    '''your code here'''
    '''***        ***'''
    S_del_rot1=noise[0]*abs(delta_rot1)+noise[1]*delta_trans
    S_del_rot2=noise[0]*abs(delta_rot2)+noise[1]*delta_trans
    S_del_trans=noise[2]*delta_trans+noise[3]*(abs(delta_rot1)+abs(delta_rot2))
    
    for p in particles:
        n_del_rot1=delta_rot1 + np.random.normal(0, S_del_rot1)
        n_del_rot2=delta_rot2 + np.random.normal(0, S_del_rot2)
        n_del_trans=delta_trans + np.random.normal(0, S_del_trans)
        
        p['history'].append([p['x'],p['y']])
        
        p['x']=p['x']+n_del_trans*np.cos(p['theta']+n_del_rot1)
        p['y']=p['y']+n_del_trans*np.sin(p['theta']+n_del_rot1)
        p['theta']=p['theta']+n_del_rot1+n_del_rot2
    return

def measurement_model(particle, landmark):
    #Compute the expected measurement for a landmark
    #and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    #calculate expected range measurement
    meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2 )
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h 
    #wrt the landmark location
    
    H = np.zeros((2,2))
    H[0,0] = (lx - px) / h[0]
    H[0,1] = (ly - py) / h[0]
    H[1,0] = (py - ly) / (h[0]**2)
    H[1,1] = (lx - px) / (h[0]**2)

    return h, H

def eval_sensor_model(sensor_data, particles):
    #Correct landmark poses with a measurement and
    #calculate particle weight

    #sensor noise
    Q_t = np.array([[0.4, 0],\
                    [0, 0.4]])

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    #update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']

        px = particle['x']
        py = particle['y']
        ptheta = particle['theta'] 

        #loop over observed landmarks 
        for i in range(len(ids)):

            #current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]
            
            #measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time
                
                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                '''your code here'''
                '''***        ***'''
                lm_x=px+meas_range*np.cos(ptheta+meas_bearing)
                lm_y=py+meas_range*np.sin(ptheta+meas_bearing)
                landmark['mu']=[lm_x, lm_y]
                
                h, H=measurement_model(particle, landmark)
                
                H_inv=np.linalg.inv(H)
                landmark['sigma']=H_inv.dot(Q_t).dot(H_inv.T)
                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above. 
                # calculate particle weight: particle['weight'] = ...
                '''your code here'''
                '''***        ***'''
                h, H=measurement_model(particle, landmark)
                
                S=landmark['sigma']
                Q=H.dot(S).dot(H.T)+Q_t
                K=S.dot(H.T).dot(np.linalg.inv(Q))
                
                delta=np.array([meas_range-h[0], angle_diff(meas_bearing, h[1])])
                
                landmark['mu']=landmark['mu']+K.dot(delta)
                landmark['sigma']=(np.identity(2) - K.dot(H)).dot(S)
                
                f=1/np.sqrt(np.linalg.det(2*math.pi*Q_t))
                exp=-0.5*np.dot(delta.T, np.linalg.inv(Q_t)).dot(delta)
                particle['weight']=particle['weight']*f*np.exp(exp)
                
    #normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer

def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.

    new_particles = []

    '''your code here'''
    '''***        ***'''

    # hint: To copy a particle from particles to the new_particles
    # list, first make a copy:
    
    u=np.random.uniform(0,(1.0/len(particles)))
    i=0
    pos=particles[0]['weight']
    
    for particle in particles:
        while u>pos:
            i+=1
            pos+=particles[i]['weight']
            
        new_particle = copy.deepcopy(particles[i])
        new_particle['weight']=1.0/len(particles)
        new_particles.append(new_particle)
        
        u+=(1.0/len(particles))



    return new_particles

def main():

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    num_particles = 100
    num_landmarks = len(landmarks)

    #create particle set
    particles = initialize_particles(num_particles, num_landmarks)

    #run FastSLAM
    for timestep in range(int(len(sensor_readings)/2)):

        #predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep,'odometry'], particles)

        #evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

        #plot filter state
        plot_state(particles, landmarks)

        #calculate new set of equally weighted particles
        particles = resample_particles(particles)

    plt.show('hold')

if __name__ == "__main__":
    main()