import numpy as np


# Mathematical form:
#  g(x;sigma,mu,theta) = 1/(sigma * sqrt(2 pi)) * exp(-(x-mu)^2 / (2 sigma^2))
#                        * exp(i theta)
#   FWHM = 2 sqrt(2 ln(2)) * sigma.
#   alpha = 1/(2 sigma^2)
#   zeta = alpha_1 + alpha_2 for two Gaussians
#   xi = 1/(2 zeta)
#   d = mu_1 - mu_2 for two Gaussians
# A list of Gaussian functions.
class Gaussian():
    def __init__(self, pos, stddev, orien, initialize):

        # Create instance variables that define the Gaussians.
        self.mu = np.array(pos)
        self.sigma = np.array(stddev)
        self.theta = np.array(orien)

        # Prepare the Gaussians for use in the integral subroutine.
        if (initialize == 1):
            self.update_integration_variables()


    def update_integration_variables(self):
        self.alpha = 0.5 / self.sigma**2
        self.cos_theta = np.cos(self.theta.imag)


    def integral(self, g):
        # Get the real parts of the integral of the product of the self and
        #   g (given) Gaussians.
        #   I(G1,G2) = Integral(Re(G1)*Re(G2) dx; -infinity..+infinity)
        #   I(G1,G2) = (pi/zeta)^1.5 * exp(-xi * d^2)
        #              * cos(theta_1) * cos(theta_2)
        one_over_zeta = 1.0 / (self.alpha + g.alpha)
        xi = 0.5 * one_over_zeta
        dist = self.mu - g.mu
        exp = np.exp(-xi * dist**2)
        intg = (np.pi * one_over_zeta)**1.5 * exp \
                * self.cos_theta * g.cos_theta
        return intg


    def accumulate(self, gaussian):
        self.mu += gaussian.mu
        self.sigma += gaussian.sigma
        self.theta += gaussian.theta


    def average(self, total_count):
        self.mu /= total_count
        self.sigma /= total_count
        self.theta /= total_count


def compute_overlap(pos_1, stddv_1, orien_1, pos_2, stddv_2, orien_2):

    # Integrate the Gaussian arrays.
    alpha_1 = 1.0 / (2.0 * stddv_1**2)
    alpha_2 = 1.0 / (2.0 * stddv_2**2)
    zeta = alpha_1 + alpha_2
    integral = np.pi
