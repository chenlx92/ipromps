#!/usr/bin/python
# Filename: ipromplib_imu_emg_pose.py

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import scipy.linalg
import scipy.stats as stats
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal as mvn

class NDProMP(object):
    """
    n-dimensional ProMP
    """
    def __init__(self, num_joints, num_basis=11, sigma_basis=0.05, num_samples=101):
        """
        :param num_joints: Number of underlying ProMPs
        :param num_basis:
        :param sigma_basis:
        :param num_samples:
        """
        if num_joints < 1:
            raise ValueError("You must declare at least 1 joint in a NDProMP")
        self.num_joints = num_joints
        self.num_basis = num_basis
        self.promps = [ProMP(num_basis, sigma_basis, num_samples) for joint in range(num_joints)]

        self.num_samples = num_samples

        self.demo_W_full = np.array([]) # the weight for each demonstration
        self.mean_W_full = np.array([])
        self.cov_W_full = np.array([])

        self.mean_W_full_updated = np.array([]) # the updated weight distribution
        self.cov_W_full_updated = np.array([])

        self.viapoints = []

    def obs_matrix(self, t):
        """
        Get the observation matrix with missing observations
        :param t: the specific time
        :return: the observation matrix
        """
        H = np.exp(-.5 * (np.array(map(lambda x: x - self.C,
                      np.tile(t, (self.num_basis, 1)).T)).T**2 / (self.sigma_basis**2)))
        zero_entry = np.zeros([1, self.num_basis])

        H_full = np.array([]).reshape(0,0)
        for idx_obs in range(self.num_joints):
            H_full = scipy.linalg.block_diag(H_full, H.T)
        return  H_full

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

        self.demo_W_full = np.array([]).reshape(self.num_demos,0)
        for idx_promp in range(self.num_joints):
            self.demo_W_full = np.hstack([self.demo_W_full, self.promps[idx_promp].W])

        self.mean_W_full = np.mean(self.demo_W_full,0)
        self.mean_W_full = self.mean_W_full.reshape([self.num_joints*self.num_basis, 1])

        self.cov_W_full = np.cov(self.demo_W_full.T) if self.num_Traj > 1 else None
        self.cov_W_full = self.cov_W_full.reshape([self.num_joints*self.num_basis, self.num_joints*self.num_basis]) if self.num_Traj > 1 else None


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
    def C(self):
        return self.promps[0].C

    @property
    def sigma_basis(self):
        return self.promps[0].sigma_basis

    @property
    def Phi(self):
        return self.promps[0].Phi

    @property
    def num_Traj(self):
        return self.promps[0].num_Traj

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

    def viapoint(self, key, viapoint=0, promp=0):
        return self.promps[promp].viapoints[viapoint][key]

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
        self.viapoints.append({"t": t, "obsy": obsys, "sigmay": sigmay})
        # we need to updated the mean anf cov of each ProMP
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
        measurement_noise = np.eye(self.num_joints)*self.promps[0].viapoints[0]['sigmay']
        new_mean_w_full = self.mean_W_full
        new_cov_w_full = self.cov_W_full

        for num_viapoint, viapoint in enumerate(self.viapoints):
            H_full = self.obs_matrix(viapoint["t"])
            y_observed = viapoint["obsy"].reshape([self.num_joints, 1])

            aux = measurement_noise + np.dot( H_full, np.dot(new_cov_w_full,H_full.T) )
            K = np.dot( np.dot(new_cov_w_full,H_full.T), np.linalg.inv(aux) )
            new_mean_w_full = new_mean_w_full + np.dot(K, y_observed - np.dot(H_full,new_mean_w_full))
            new_cov_w_full = new_cov_w_full - np.dot(K, np.dot(H_full,new_cov_w_full))

        for i in range(self.num_joints):
            self.promps[i].meanW_updated = new_mean_w_full.reshape([self.num_joints,self.num_basis]).T[:,i]
            self.promps[i].sigmaW_updated = new_cov_w_full[i*self.num_basis:(1+i)*self.num_basis, i*self.num_basis:(i+1)*self.num_basis]

        trajectory = np.dot( self.Phi.T, new_mean_w_full.reshape([self.num_joints,self.num_basis]).T )

        return trajectory


class ProMP(object):
    """
    Uni-dimensional probabilistic MP
    """
    def __init__(self, num_basis=11, sigma_basis=0.05, num_samples=101):
        self.x = np.linspace(0.0, 1.0, num_samples) # the time value
        self.num_samples = num_samples    # num of samples
        self.num_basis = num_basis          # num of basis func
        self.sigmaSignal = float('inf') # the zero-mean noise, including modelling error and the system noise
        self.sigma_basis = sigma_basis              # the sigma of basis func
        self.C = np.arange(0,num_basis)/(num_basis-1.0)     # the mean of basis func along the time
        self.Phi = np.exp(-.5 * (np.array(map(lambda x: x - self.C, np.tile(self.x, (self.num_basis, 1)).T)).T ** 2 / (self.sigma_basis ** 2)))

        self.viapoints = [] # the via point list
        self.W = np.array([]) # the weight for each demon
        self.num_Traj = 0     # the demon number
        self.Y = np.empty((0, self.num_samples), float)   # the demon traj array

        # the w prior distribution
        self.meanW = None
        self.sigmaW = None

        # the updated distribution from unit promp
        self.meanW_unit = None
        self.sigmaW_unit = None

        # the param of output distribution from ndpromp with via point and correlation amongs joints
        self.meanW_updated = None
        self.sigmaW_updated = None

        # the scaling factor for letting each traj have same duration
        self.alpha_demo = []

    def add_demonstration(self, demonstration):
        interpolate = interp1d(np.linspace(0, 1, len(demonstration)), demonstration, kind='cubic')
        stretched_demo = interpolate(self.x)
        self.Y = np.vstack((self.Y, stretched_demo))
        self.num_Traj = len(self.Y)
        self.W = np.dot(np.linalg.inv(np.dot(self.Phi, self.Phi.T)), np.dot(self.Phi, self.Y.T)).T  # weights for each trajectory, MLE here
        self.meanW = np.mean(self.W, 0)                                                             # mean of weights
        # w1 = np.array(map(lambda x: x - self.meanW.T, self.W))
        # self.sigmaW = np.dot(w1.T, w1)/self.num_Traj              # covariance of weights
        self.sigmaW = np.cov((self.W).T) if self.num_Traj>1 else None
        self.sigmaSignal = np.sum(np.sum((np.dot(self.W, self.Phi) - self.Y) ** 2)) / (self.num_Traj*self.num_samples)

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
            PhiT = np.exp(-.5 * (np.array(map(lambda x: x - self.C, np.tile(viapoint['t'], (11, 1)).T)).T ** 2 / (self.sigma_basis ** 2)))

            # Conditioning
            aux = viapoint['sigmay'] + np.dot(np.dot(PhiT.T, newSigma), PhiT)
            K = np.dot(newSigma, PhiT) * 1 / aux
            newMu = newMu + np.dot(K, (viapoint['obsy'] - np.dot(PhiT.T, newMu)))  # new weight mean conditioned on observations
            newSigma = newSigma - np.dot(K, np.dot(PhiT.T, newSigma))

        self.meanW_unit = newMu
        self.sigmaW_unit = newSigma

        sampW = np.random.multivariate_normal(newMu, randomness*newSigma, 1).T
        return np.dot(self.Phi.T, sampW)

    def plot_prior(self, x=None, legend='', color='b'):
        """
        plot the prior distribution from training sets
        """
        mean = np.dot(self.Phi.T, self.meanW)
        x = self.x if x is None else x
        plt.plot(x, mean, color=color, label=legend, linewidth=3, alpha=0.4)
        std = self.get_std()
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.6)

    def plot_unit(self, x=None, legend='', color='b'):
        """
        plot the unit update ProMP, only valid from unit ProMP
        """
        mean = np.dot(self.Phi.T, self.meanW_unit)
        x = self.x if x is None else x
        plt.plot(x, mean, color=color, label=legend)
        std = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW_unit, self.Phi))))
        plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.4)

    def plot_updated(self, x=None, legend='', color='b', via_show=True):
        """
        plot the updated distribution, valid from NDProMP or IProMP
        """
        # if self.meanW_updated==None:
        #     print "there is no updated para from NDProMP or IProMP"
        #     return
        mean0 = np.dot(self.Phi.T, self.meanW_updated)
        std0 = 2 * np.sqrt(np.diag(np.dot(self.Phi.T, np.dot(self.sigmaW_updated, self.Phi))))
        x = self.x if x is None else x
        plt.plot(x, mean0, linestyle='--', color=color, label=legend, linewidth=5)
        plt.fill_between(x, mean0-std0, mean0+std0, color=color, alpha=0.4)

        if via_show == True:
            for viapoint_id, viapoint in enumerate(self.viapoints):
                x_index = x[int(round((len(x)-1)*viapoint['t'], 0))]
                # plt.plot(x_index, viapoint['obsy'], marker="o", markersize=10, label="Observation {}".format(viapoint_id), color=color)
                plt.plot(x_index, viapoint['obsy'], marker="o", markersize=10, color=color)


class IProMP(NDProMP):
    """
    (n)-dimensional Interaction ProMP, derived from NDProMP
    """
    def __init__(self, num_joints=19, num_basis=11, sigma_basis=0.05, num_samples=101, num_obs_joints=None):
        """
        construct function, call NDProMP construct function and define the member variables
        """
        NDProMP.__init__(self, num_joints=num_joints, num_basis=num_basis, sigma_basis=sigma_basis, num_samples=num_samples)

        self.viapoints = []
        self.num_obs_joints = num_obs_joints    # the observed joints number

        # the scaling factor
        self.alpha_demo = []
        self.alpha_mean = []
        self.alpha_std = []

    def obs_matrix(self, t):
        """
        Get the observation matrix with missing observations
        :param t: the specific time
        :return: the observation matrix
        """
        H = np.exp(-.5 * (np.array(map(lambda x: x - self.C,
                      np.tile(t, (self.num_basis, 1)).T)).T ** 2 / (self.sigma_basis ** 2)))
        zero_entry = np.zeros([1, self.num_basis])

        H_full = np.array([]).reshape(0,0)
        for idx_obs in range(self.num_obs_joints):
            H_full = scipy.linalg.block_diag(H_full, H.T)
        for idx_non_obs in range(self.num_joints - self.num_obs_joints):
            H_full = scipy.linalg.block_diag(H_full, zero_entry)
        return  H_full


    def add_viapoint(self, t, obsys, sigmay):
        """
        Add a viapoint i.e. an observation at a specific time
        :param t: Time of observation
        :param obsys: List of observations obys[joint] for each joint
        :param sigmay:
        :return:
        """
        if len(obsys) != self.num_joints:
            raise ValueError("The given viapoint has {} joints while num_joints={}".format(len(obsys), self.num_joints))
        # save the viapoint to ProMP and IProMP
        for joint_demo in range(self.num_joints):
            self.promps[joint_demo].add_viapoint(t, obsys[joint_demo], sigmay)
        self.viapoints.append({"t": t, "obs": obsys, "sigmay": sigmay})
        # the prior
        new_mean_w_full = self.mean_W_full
        new_cov_w_full = self.cov_W_full
        for num_viapoint in range(self.num_viapoints):
            H_full = self.obs_matrix(self.viapoint("t",num_viapoint))
            # the observation of specific time
            y_observed = np.array([]).reshape(0,1)
            for idx_promp in range(self.num_joints):
                y_observed = np.vstack([y_observed, self.promps[idx_promp].viapoints[num_viapoint]['obsy']])
            # update the distribution
            aux = sigmay + np.dot(H_full, np.dot(new_cov_w_full, H_full.T))
            K = np.dot(np.dot(new_cov_w_full, H_full.T), np.linalg.inv(aux))
            new_mean_w_full = new_mean_w_full + np.dot(K, y_observed - np.dot(H_full, new_mean_w_full))
            new_cov_w_full = new_cov_w_full - np.dot(K, np.dot(H_full, new_cov_w_full))
        # save the updated result
        self.mean_W_full_updated = new_mean_w_full
        self.cov_W_full_updated = new_cov_w_full
        # set the theta of each channel
        for i in range(self.num_joints):
            self.promps[i].meanW_updated = new_mean_w_full.reshape([self.num_joints, self.num_basis]).T[:, i]
            self.promps[i].sigmaW_updated = new_cov_w_full[i * self.num_basis:(1 + i) * self.num_basis, i * self.num_basis:(i + 1) * self.num_basis]


    def generate_trajectory(self, randomness=1e-10):
        """
        Add a viapoint i.e. an observation at a specific time
        :param randomness: the measurement noise
        :return: the mean of predictive distribution
        """
        new_mean_w_full = self.mean_W_full_updated
        trajectory = np.dot( self.Phi.T, new_mean_w_full.reshape([self.num_joints,self.num_basis]).T )
        return trajectory

    def prob_obs(self):
        """
        compute the pdf of observation sets
        :return: the total joint probability
        """
        prob_full = 0.0
        for idx_obs, obs in enumerate(self.viapoints):
            H_full = self.obs_matrix(obs["t"])
            # the y mean and cov
            mean_t = np.dot(H_full, self.mean_W_full)[:,0]
            cov_t = np.dot(H_full, np.dot(self.cov_W_full, H_full.T)) + obs["sigmay"]
            # compute the log likelihood
            prob = mvn.pdf(obs["obs"], mean_t, cov_t)
            log_pro = math.log(prob) if prob !=0.0 else -np.inf
            prob_full = prob_full + log_pro
        return prob_full

    def add_alpha(self, alpha):
        """
        Add a phase to the trajectory
        Observations and corresponding basis activations
        :param t: timestamp of viapoint
        :param obsy: observed value at time t
        :param sigmay: observation variance (constraint strength)
        :return:
        """
        self.alpha_demo.append(alpha)
        self.alpha_mean = np.mean(self.alpha_demo)
        self.alpha_std = np.std(self.alpha_demo)

    def alpha_candidate(self, num):
        """
        compute the alpha candidate by unit sampling
        Observations and corresponding basis activations
        :param num: the num of alpha candidate
        :return: the list of alpha candidate
        """
        alpha_candidate = np.linspace(self.alpha_mean-2*self.alpha_std, self.alpha_mean+2*self.alpha_std, num)
        candidate_pdf = stats.norm.pdf(alpha_candidate, self.alpha_mean, self.alpha_std)
        alpha_gen = {'candidate': alpha_candidate, 'prob': candidate_pdf}
        return alpha_gen

    def alpha_log_likelihood(self, alpha_candidate, obs, rate, sigmay):
        prob_full = 0.0
        for obs_idx in range(len(obs)):
            PhiT = np.exp(-.5 * (np.array(map(lambda x: x-self.C, np.tile(obs_idx/rate/alpha_candidate,
                           (self.num_basis, 1)).T)).T**2 / (self.sigma_basis**2)))
            zero_term = np.zeros((1,len(PhiT)))
            A = scipy.linalg.block_diag(PhiT.T, PhiT.T, PhiT.T, PhiT.T,
                                        PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T, PhiT.T,
                                        zero_term, zero_term, zero_term, zero_term, zero_term, zero_term, zero_term)
            # A = self.obs_matrix(obs["t"])
            mean_t = np.dot(A, self.mean_W_full)[:,0]
            cov_t = np.dot(np.dot(A, self.cov_W_full),  A.T) + sigmay

            prob = mvn.pdf(obs[obs_idx], mean_t, cov_t)
            log_prob = math.log(prob) if prob !=0.0 else -np.inf
            prob_full = prob_full + log_prob
        return prob_full

    def alpha_estimate(self, alpha_candidate, obs, rate, sigmay):
        pp_list = []
        for idx in range(len(alpha_candidate['candidate'])):
            lh = self.alpha_log_likelihood(alpha_candidate['candidate'][idx], obs, rate, sigmay)
            pp = math.log(alpha_candidate['prob'][idx]) + lh
            pp_list.append(pp)
        id_max = np.argmax(pp_list)
        return  id_max

    def gen_predict_traj(self, alpha, rate):
        traj = self.generate_trajectory()
        points = self.x
        grid = np.linspace(0, 1.0, np.int(alpha * rate))
        predict_traj = griddata(points, traj, grid, method='linear')
        id_predict_traj = np.linspace(0.0, alpha, len(grid))
        return id_predict_traj, predict_traj

