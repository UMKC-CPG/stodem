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
        # Normalization constant: sqrt(I(G,G)) = (pi*sigma^2)^0.75 * |cos_theta|
        # Cached here so integral() pays only 1 mul + 1 div per call.
        self.self_norm = (np.pi * self.sigma**2)**0.75 \
                         * np.abs(self.cos_theta)


    def integral(self, g):
        # Normalized overlap integral, bounded to [-1, +1].
        #   I_norm(G1,G2) = I_raw(G1,G2) / sqrt(I(G1,G1) * I(G2,G2))
        #   I_raw(G1,G2)  = (pi/zeta)^1.5 * exp(-xi * d^2)
        #                   * cos(theta_1) * cos(theta_2)
        # Returns 0 when either Gaussian is fully apathetic (cos_theta=0).
        one_over_zeta = 1.0 / (self.alpha + g.alpha)
        xi = 0.5 * one_over_zeta
        dist = self.mu - g.mu
        exp = np.exp(-xi * dist**2)
        raw = (np.pi * one_over_zeta)**1.5 * exp \
              * self.cos_theta * g.cos_theta
        norm = self.self_norm * g.self_norm
        return np.where(norm > 0.0, raw / norm, 0.0)


    def accumulate(self, gaussian):
        self.mu += gaussian.mu
        self.sigma += gaussian.sigma
        self.theta += gaussian.theta


    def average(self, total_count):
        self.mu /= total_count
        self.sigma /= total_count
        self.theta /= total_count
