#!/usr/bin/python
# Filename: ipromp.py

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.stats import multivariate_normal as mvn

class NDProMP(object):
    """
    n-dimensional ProMP
    """
    def __init__(self, num_joints, nrBasis=11, sigma=0.05, num_samples=101):
        """
        :param num_joints: Number of underlying ProMPs
        :param nrBasis:
        :param sigma:
        """
        if num_joints < 1:
            raise ValueError("You must declare at least 1 joint in a NDProMP")
        self.num_joints = num_joints
        self.nrBasis = nrBasis
        self.promps = [ProMP(nrBasis, sigma, num_samples) for joint in range(num_joints)]
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'chocolate', 'deepskyblue', 'sage', 'darkviolet', 'crimson']
        
        self.num_samples = num_samples
        self.demo_W_full = np.array([])
        self.mean_W_full = np.array([])
        self.cov_W_full = np.array([])
        
    @property
    def x(self):
        return self.promps[0].x

    def add_demonstration(self, demonstration):
        """
        Add a new N-joints demonstration[time][joint] and update the model
        :param demonstration: List of "num_joints" demonstrations
        :return:
        """        
        demonstration = np.array(demonstration).T  # Revert the representation for each time for each joint, for each joint for each time

        if len(demonstration) != self.num_joints:
            raise ValueError("The given demonstration has {} joints while num_joints={}".format(len(demonstration), self.num_joints))

        for joint_demo_idx, joint_demo in enumerate(demonstration):
            self.promps[joint_demo_idx].add_demonstration(joint_demo)
        
        self.demo_W_full = np.hstack([self.promps[0].W, self.promps[1].W, self.promps[2].W, self.promps[3].W, self.promps[4].W, 
                                      self.promps[5].W, self.promps[6].W, self.promps[7].W, self.promps[8].W, self.promps[9].W, 
                                        self.promps[10].W,self.promps[11].W,self.promps[12].W, self.promps[13].W, self.promps[14].W])
        
        self.mean_W_full = np.mean(self.demo_W_full,0)
        self.mean_W_full = self.mean_W_full.reshape([self.num_joints*self.nrBasis, 1])
        
        self.cov_W_full = np.cov(self.demo_W_full.T) if self.promps[0].nrTraj > 1 else None
        self.cov_W_full = self.cov_W_full.reshape([self.num_joints*self.nrBasis, self.num_joints*self.nrBasis]) if self.promps[0].nrTraj > 1 else None


    @property
    def num_demos(self):
        return self.promps[0].num_demos

    @property
    def num_points(self):
        return self.promps[0].num_points

    @property
    def num_viapoints(self):
        return self.promps[0].num_viapoints

    @property
    def goal_bounds(self):
        return [joint.goal_bounds for joint in self.promps]

    @property
    def goal_means(self):
        return [joint.goal_mean for joint in self.promps]

    def get_bounds(self, t):
        """
        Return the bounds of all joints at time t
        :param t: 0 <= t <= 1
        :return: [(lower boundary joints 0, upper boundary joints 0), (lower boundary joint 1), upper)...]
        """
        return [joint.get_bounds(t) for joint in self.promps]

    def get_means(self, t):
        """
        Return the mean of all joints at time t
        :param t: 0 <= t <= 1
        :return: [mean joint 1, mean joint 2, ...]
        """
        return [joint.get_mean(t) for joint in self.promps]

    def get_stds(self):
        """
        Return the standard deviation of all joints
        :param t: 0 <= t <= 1
        :return: [std joint 1, std joint 2, ...]
        """
        return [joint.get_std() for joint in self.promps]

    def clear_viapoints(self):
        for promp in self.promps:
            promp.clear_viapoints()

#    def add_viapoint(self, t, obsys, sigmay=[]):
    def add_viapoint(self, t, obsys, sigmay=1e-6):
        """
        Add a viapoint i.e. an observation at a specific time
        :param t: Time of observation
        :param obsys: List of observations obys[joint] for each joint
        :param sigmay:
        :return:
        """
        if len(obsys) != self.num_joints:
            raise ValueError("The given viapoint has {} joints while num_joints={}".format(len(obsys), self.num_joints))
            
        for joint_demo in range(self.num_joints):
#            self.promps[joint_demo].add_viapoint(t, obsys[joint_demo], sigmay[joint_demo])
            self.promps[joint_demo].add_viapoint(t, obsys[joint_demo], sigmay)
            
        self.generate_trajectory()
            

    def set_goal(self, obsy, sigmay=1e-6):
        if len(obsy) != self.num_joints:
            raise ValueError("The given goal state has {} joints while num_joints={}".format(len(obsy), self.num_joints))

        for joint_demo in range(self.num_joints):
            self.promps[joint_demo].set_goal(obsy[joint_demo], sigmay)

    def set_start(self, obsy, sigmay=1e-6):
        if len(obsy) != self.num_joints:
            raise ValueError("The given start state has {} joints while num_joints={}".format(len(obsy), self.num_joints))

        for joint_demo in range(self.num_joints):
            self.promps[joint_demo].set_start(obsy[joint_demo], sigmay)

    def generate_trajectory(self, randomness=1e-10):
        
        trajectory = []
        
        measurement_noise = np.eye(self.num_joints)*self.promps[0].viapoints[0]['sigmay']
        new_mean_w_full = self.mean_W_full
        new_cov_w_full = self.cov_W_full
        PhiT_full = np.array([])
        
        for num_viapoint in range( len(self.promps[0].viapoints) ):
            PhiT = np.exp(-.5 * (np.array(map(lambda x: x - self.promps[0].C, np.tile(self.promps[0].viapoints[num_viapoint]['t'], (self.nrBasis, 1)).T)).T ** 2 / (self.promps[0].sigma ** 2)))
            PhiT = PhiT / sum(PhiT)  # basis functions at observed time points            
            # here is a trick for construct the observation matrix            
            PhiT_full = scipy.linalg.block_diag(PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, 
                                                PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T)

            y_observed = np.array( [[self.promps[0].viapoints[num_viapoint]['obsy']], [self.promps[1].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[2].viapoints[num_viapoint]['obsy']], [self.promps[3].viapoints[num_viapoint]['obsy']],
                                    [self.promps[4].viapoints[num_viapoint]['obsy']], [self.promps[5].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[6].viapoints[num_viapoint]['obsy']], [self.promps[7].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[8].viapoints[num_viapoint]['obsy']], [self.promps[9].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[10].viapoints[num_viapoint]['obsy']], [self.promps[11].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[12].viapoints[num_viapoint]['obsy']], [self.promps[13].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[14].viapoints[num_viapoint]['obsy']]] )
            
            aux = measurement_noise + np.dot( PhiT_full, np.dot(new_cov_w_full,PhiT_full.T) )
            K = np.dot( np.dot(new_cov_w_full,PhiT_full.T), np.linalg.inv(aux) )
            
            new_mean_w_full = new_mean_w_full + np.dot(K, y_observed - np.dot(PhiT_full,new_mean_w_full))
            new_cov_w_full = new_cov_w_full + np.dot(K, np.dot(PhiT_full,new_cov_w_full))
        
        for i in range(15):
            self.promps[i].meanW_updated = new_mean_w_full.reshape([self.num_joints,self.nrBasis]).T[:,i]
            self.promps[i].sigmaW_updated = new_cov_w_full[i*self.nrBasis:(1+i)*self.nrBasis, i*self.nrBasis:(i+1)*self.nrBasis]
        
        trajectory = np.dot( self.promps[0].Phi.T, new_mean_w_full.reshape([self.num_joints,self.nrBasis]).T )
        
        return trajectory
        

    def plot(self, x=None, joint_names=(), output_randomess=1e-105):
        """
        Plot the means and variances of gaussians, requested viapoints as well as an output trajectory (dotted)
        :param output_randomess: 0. to 1., -1 to disable output plotting
        """
        if output_randomess >= 0:
            output = self.generate_trajectory(output_randomess).T

        for promp_idx, promp in enumerate(self.promps):
            color = self.colors[promp_idx % len(self.colors)]
            joint_name = "Joint {}".format(promp_idx+1) if len(joint_names) == 0 else joint_names[promp_idx]
            promp.plot(x, joint_name, color)
            if output_randomess >= 0:
                plt.plot(x, output[promp_idx], linestyle='--', label="Out {}".format(joint_name), color=color, lw=5)
        
#        mean0 = np.dot(self.promps[0].Phi.T, self.promps[0].meanW_updated)
#        std0 = 2 * np.sqrt(np.diag(np.dot(self.promps[0].Phi.T, np.dot(self.promps[0].sigmaW_updated, self.promps[0].Phi))))
#        plt.fill_between(x, mean0-std0, mean0 + std0, color='r', alpha=0.4)
#        
#        mean1 = np.dot(self.promps[1].Phi.T, self.promps[1].meanW_updated)
#        std1 = 2 * np.sqrt(np.diag(np.dot(self.promps[1].Phi.T, np.dot(self.promps[1].sigmaW_updated, self.promps[1].Phi))))
#        plt.fill_between(x, mean1-std1, mean1 + std1, color='b', alpha=0.4)

    def plot_updated(self, x=None, output_randomess=1e-105):
        mean0 = np.dot(self.promps[0].Phi.T, self.promps[0].meanW_updated)
        std0 = 2 * np.sqrt(np.diag(np.dot(self.promps[0].Phi.T, np.dot(self.promps[0].sigmaW_updated, self.promps[0].Phi))))
        plt.fill_between(x, mean0-std0, mean0 + std0, color='r', alpha=0.4)
        
        mean1 = np.dot(self.promps[1].Phi.T, self.promps[1].meanW_updated)
        std1 = 2 * np.sqrt(np.diag(np.dot(self.promps[1].Phi.T, np.dot(self.promps[1].sigmaW_updated, self.promps[1].Phi))))
        plt.fill_between(x, mean1-std1, mean1 + std1, color='b', alpha=0.4)
        
        
        
class ProMP(object):
    """
    Uni-dimensional probabilistic MP
    """
    def __init__(self, nrBasis=11, sigma=0.05, num_samples=101):
        self.x = np.linspace(0, 1, num_samples) # the time value
        self.nrSamples = len(self.x)    # num of samples
        self.nrBasis = nrBasis          # num of basis func
        self.sigmaSignal = float('inf') # the zero-mean noise, including modelling error and the system noise
        self.sigma = sigma              # the sigma of basis func
        self.C = np.arange(0,nrBasis)/(nrBasis-1.0) # the mean of basis func along the time
        self.Phi = np.exp(-.5 * (np.array(map(lambda x: x - self.C, np.tile(self.x, (self.nrBasis, 1)).T)).T ** 2 / (self.sigma ** 2)))
        self.Phi /= sum(self.Phi)       # the basis func vector at diff time, an (baisi func)11 X (sample len)100 matrix

        self.viapoints = []
        self.W = np.array([])
        self.nrTraj = 0
        self.Y = np.empty((0, self.nrSamples), float)   # the traj along time
        
        # the w probability distribution from training sets        
        self.meanW = None
        self.sigmaW = None
        
        # the param of output distribution from unit promp with the via points
        self.meanW_via = None
        self.sigmaW_via = None
        
        # the param of output distribution from ndpromp with via point and correlation amongs joints
        self.meanW_updated = None
        self.sigmaW_updated = None

    def add_demonstration(self, demonstration):
        interpolate = interp1d(np.linspace(0, 1, len(demonstration)), demonstration, kind='cubic')
        stretched_demo = interpolate(self.x)
        self.Y = np.vstack((self.Y, stretched_demo))
        self.nrTraj = len(self.Y)
        self.W = np.dot(np.linalg.inv(np.dot(self.Phi, self.Phi.T)), np.dot(self.Phi, self.Y.T)).T  # weights for each trajectory, MLE here
        self.meanW = np.mean(self.W, 0)                                                             # mean of weights
        # w1 = np.array(map(lambda x: x - self.meanW.T, self.W))
        # self.sigmaW = np.dot(w1.T, w1)/self.nrTraj              # covariance of weights
        self.sigmaW = np.cov((self.W).T) if self.nrTraj>1 else None
        self.sigmaSignal = np.sum(np.sum((np.dot(self.W, self.Phi) - self.Y) ** 2)) / (self.nrTraj * self.nrSamples)
        
    @property
    def noise(self):
        return self.sigmaSignal

    @property
    def num_demos(self):
        return self.Y.shape[0]

    @property
    def num_points(self):
        return self.Y.shape[1]

    @property
    def num_viapoints(self):
        return len(self.viapoints)

    @property
    def goal_bounds(self):
        """
        Joint boundaries of the last point
        :return: (lower boundary, upper boundary)
        """
        return self._get_bounds(-1)

    @property
    def goal_mean(self):
        """
        Mean of the last point
        :return: scalar
        """
        return self._get_mean(-1)

    def get_bounds(self, t):
        """
        Return the bounds at time t
        :param t: 0 <= t <= 1
        :return: (lower boundary, upper boundary)
        """
        return self._get_bounds(int(self.num_points*t))

    def get_mean(self, t):
        """
        Return the mean at time t
        :param t: 0 <= t <= 1
        :return: scalar
        """
        return self._get_mean(int(self.num_points*t))

    def get_std(self):
        std = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW, self.Phi))))
        return std

    def _get_mean(self, t_index):
        mean = np.dot(self.Phi.T, self.meanW)
        return mean[t_index]

    def _get_bounds(self, t_index):
        mean = self._get_mean(t_index)
        std = self.get_std()
        return mean - std, mean + std

    def clear_viapoints(self):
        del self.viapoints[:]

    def add_viapoint(self, t, obsy, sigmay=1e-6):
        """
        Add a viapoint to the trajectory
        Observations and corresponding basis activations
        :param t: timestamp of viapoint
        :param obsy: observed value at time t
        :param sigmay: observation variance (constraint strength)
        :return:
        """
        self.viapoints.append({"t": t, "obsy": obsy, "sigmay": sigmay})

    def set_goal(self, obsy, sigmay=1e-6):
        self.add_viapoint(1., obsy, sigmay)

    def set_start(self, obsy, sigmay=1e-6):
        self.add_viapoint(0., obsy, sigmay)

    def generate_trajectory(self, randomness=1e-10):
        """
        Outputs a trajectory
        :param randomness: float between 0. (output will be the mean of gaussians) and 1. (fully randomized inside the variance)
        :return: a 1-D vector of the generated points
        """
        newMu = self.meanW
        newSigma = self.sigmaW

        for viapoint in self.viapoints:
            PhiT = np.exp(-.5 * (np.array(map(lambda x: x - self.C, np.tile(viapoint['t'], (11, 1)).T)).T ** 2 / (self.sigma ** 2)))
            PhiT = PhiT / sum(PhiT)  # basis functions at observed time points
            
            # Conditioning
            aux = viapoint['sigmay'] + np.dot(np.dot(PhiT.T, newSigma), PhiT)
            K = np.dot(newSigma, PhiT) * 1 / aux
            newMu = newMu + np.dot(K, (viapoint['obsy'] - np.dot(PhiT.T, newMu)))  # new weight mean conditioned on observations
            newSigma = newSigma - np.dot(K, np.dot(PhiT.T, newSigma))
            
        self.meanW_via = newMu
        self.sigmaW_via = newSigma
        
        sampW = np.random.multivariate_normal(newMu, randomness*newSigma, 1).T
        return np.dot(self.Phi.T, sampW)


    def plot(self, x=None, legend='', color='b'):
        """
        plot the prior distribution from training sets
        """
        mean = np.dot(self.Phi.T, self.meanW)
        x = self.x if x is None else x
        # plt.plot(x, mean, color=color, label=legend, linewidth=3)
#        std = self.get_std()
        std = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW, self.Phi))))
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.4)
        for viapoint_id, viapoint in enumerate(self.viapoints):
            x_index = x[int(round((len(x)-1)*viapoint['t'], 0))]
            plt.plot(x_index, viapoint['obsy'], marker="o", markersize=10, label="Via {} {}".format(viapoint_id, legend), color=color)
    
    def plot_unit(self, x=None, legend='', color='b'):
        """
        plot the unit update ProMP, only valid from unit ProMP
        """
        mean = np.dot(self.Phi.T, self.meanW_via)
        x = self.x if x is None else x
        plt.plot(x, mean, color=color, label=legend)
        std = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW_via, self.Phi))))
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.4)
#        for viapoint_id, viapoint in enumerate(self.viapoints):
#            x_index = x[int(round((len(x)-1)*viapoint['t'], 0))]
#            plt.plot(x_index, viapoint['obsy'], marker="o", markersize=10, label="Via {} {}".format(viapoint_id, legend), color=color)

    def plot_updated(self, x=None, legend='', color='b', via_show=True):
        """
        plot the updated distribution, only valid from NDProMP or IProMP
        """
        mean0 = np.dot(self.Phi.T, self.meanW_updated)
        std0 = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW_updated, self.Phi))))
        plt.plot(x, mean0, linestyle='--', color=color, label=legend, linewidth=5)        
        plt.fill_between(x, mean0-std0, mean0+std0, color=color, alpha=0.4)
        
        if via_show == True:
            for viapoint_id, viapoint in enumerate(self.viapoints):
                x_index = x[int(round((len(x)-1)*viapoint['t'], 0))]
                plt.plot(x_index, viapoint['obsy'], marker="o", markersize=10, label="Via {} {}".format(viapoint_id, legend), color=color)
            
            
class IProMP(NDProMP):
    """
    (n)-dimensional Interaction ProMP, derived from NDProMP
    """
    def __init__(self, num_joints=15, nrBasis=11, sigma=0.05, num_samples=101):
        """
        construct function, call NDProMP construct function onlu
        """
        NDProMP.__init__(self, num_joints=15, nrBasis=11, sigma=0.05, num_samples=101)
        self.obsy = []
        
        
    def add_viapoint(self, t, obsys, sigmay=1e-6):
        """
        Add a viapoint i.e. an observation at a specific time
        :param t: Time of observation
        :param obsys: List of observations obys[joint] for each joint
        :param sigmay:
        :return:
        """
        if len(obsys) != self.num_joints:
            raise ValueError("The given viapoint has {} joints while num_joints={}".format(len(obsys), self.num_joints))
        for joint_demo in range(self.num_joints):
            self.promps[joint_demo].add_viapoint(t, obsys[joint_demo], sigmay)
        self.generate_trajectory()
        
        
    def generate_trajectory(self, randomness=1e-10):
        
        trajectory = []
        
        measurement_noise = np.eye(self.num_joints)*self.promps[0].viapoints[0]['sigmay']
        new_mean_w_full = self.mean_W_full
        new_cov_w_full = self.cov_W_full
        PhiT_full = np.array([])
        
        for num_viapoint in range( len(self.promps[0].viapoints) ):
            PhiT = np.exp(-.5 * (np.array(map(lambda x: x - self.promps[0].C, np.tile(self.promps[0].viapoints[num_viapoint]['t'], (self.nrBasis, 1)).T)).T ** 2 / (self.promps[0].sigma ** 2)))
            PhiT = PhiT / sum(PhiT)  # basis functions at observed time points            
            # here is a trick for construct the observation matrix
            zero_entry = np.zeros([1,11])
            PhiT_full = scipy.linalg.block_diag(PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, 
                                                zero_entry, zero_entry, zero_entry, zero_entry, zero_entry, zero_entry, zero_entry)

            y_observed = np.array( [[self.promps[0].viapoints[num_viapoint]['obsy']], [self.promps[1].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[2].viapoints[num_viapoint]['obsy']], [self.promps[3].viapoints[num_viapoint]['obsy']],
                                    [self.promps[4].viapoints[num_viapoint]['obsy']], [self.promps[5].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[6].viapoints[num_viapoint]['obsy']], [self.promps[7].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[8].viapoints[num_viapoint]['obsy']], [self.promps[9].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[10].viapoints[num_viapoint]['obsy']], [self.promps[11].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[12].viapoints[num_viapoint]['obsy']], [self.promps[13].viapoints[num_viapoint]['obsy']], 
                                    [self.promps[14].viapoints[num_viapoint]['obsy']]] )
            
            aux = measurement_noise + np.dot( PhiT_full, np.dot(new_cov_w_full,PhiT_full.T) )
            K = np.dot( np.dot(new_cov_w_full,PhiT_full.T), np.linalg.inv(aux) )
            
            new_mean_w_full = new_mean_w_full + np.dot(K, y_observed - np.dot(PhiT_full,new_mean_w_full))
            new_cov_w_full = new_cov_w_full + np.dot(K, np.dot(PhiT_full,new_cov_w_full))
        
        for i in range(15):
            self.promps[i].meanW_updated = new_mean_w_full.reshape([self.num_joints,self.nrBasis]).T[:,i]
            self.promps[i].sigmaW_updated = new_cov_w_full[i*self.nrBasis:(1+i)*self.nrBasis, i*self.nrBasis:(i+1)*self.nrBasis]
        
        trajectory = np.dot( self.promps[0].Phi.T, new_mean_w_full.reshape([self.num_joints,self.nrBasis]).T )
        return trajectory

    def add_obsy(self, t, obsy, sigmay=1e-6):
        """
        Add a observation to the trajectory
        Observations and corresponding basis activations
        :param t: timestamp of viapoint
        :param obsy: observed value at time t
        :param sigmay: observation variance (constraint strength)
        :return:
        """
        self.obsy.append({"t": t, "obsy": obsy, "sigmay": sigmay})
            
    def prob_obs(self):
        """
        compute the pdf of observation sets
        :return: the total joint probability
        """
        PhiT = self.promps[0].Phi.T
        # here is a trick for construct the observation matrix
        PhiT_full = scipy.linalg.block_diag(PhiT, PhiT, PhiT, PhiT, PhiT, PhiT, PhiT, PhiT, PhiT, PhiT, PhiT, PhiT, PhiT, PhiT, PhiT)
        # the obsvation distribution from weight distribution
        mean = np.dot( PhiT_full, self.mean_W_full )
        cov = np.dot( PhiT_full, np.dot(self.cov_W_full, PhiT_full.T))
        # compute the pdf of obsy
        prob_full = 1.0
        for obsy in self.obsy:
            # the mean for EMG singnal observation distribution
            mean_t = mean.reshape(self.num_joints, self.num_samples).T[np.int(obsy['t']*100),:]
            mean_t = mean_t[0:8]
            # the covariance for EMG singnal observation distribution
            idx = np.arange(15)*101 + np.int(obsy['t']*100)
            cov_t = cov[idx,:][:,idx]
            cov_t = cov_t[0:8,0:8]            
            # compute the prob of partial observation
            prob = mvn.pdf(obsy['obsy'][0:8], mean_t, cov_t)
            prob_full = prob_full*prob
        return prob_full
