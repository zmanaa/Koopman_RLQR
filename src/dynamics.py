import scipy.linalg as la
import numpy as np


class Satellite:
    def __init__(self, inertia_matrix, initial_omega):
        """
        Initializes the Satellite.

        Parameters:
        - inertia_matrix (np.ndarray): Inertia matrix J (3x3).
        - initial_omega (np.ndarray): Initial angular velocity vector (3,).
        """
        self.J = inertia_matrix
        self.omega = initial_omega

    def get_angular_velocity(self):
        return self.omega

    def set_angular_velocity(self, new_omega):
        self.omega = new_omega


class Dynamics:
    def __init__(self, satellite):
        """
        Initializes the Dynamics.

        Parameters:
        - satellite (Satellite): Satellite object.
        """
        self.satellite = satellite

    def compute_dynamics(self, omega, torque):
        """
        Computes the derivative of the angular velocity (omega_dot) based on the current state.
        This is the function f(omega, t) for the RK4 method.

        Parameters:
        - omega (np.ndarray): Current angular velocity vector (3,).
        - torque (np.ndarray): Current torque vector (3,).

        Returns:
        - omega_dot (np.ndarray): Derivative of angular velocity.
        """
        J = self.satellite.J
        omega_dot = la.inv(J) @ (-np.cross(omega, J @ omega) + torque)
        return omega_dot

    def step(self, torque, dt):
        """
        Evolves the angular velocity using the Runge-Kutta 4th order (RK4) method.

        Parameters:
        - torque (np.ndarray): Applied torque vector (3,).
        - dt (float): Time step.
        """
        omega = self.satellite.get_angular_velocity()

        k1 = self.compute_dynamics(omega, torque)
        k2 = self.compute_dynamics(omega + 0.5 * dt * k1, torque)
        k3 = self.compute_dynamics(omega + 0.5 * dt * k2, torque)
        k4 = self.compute_dynamics(omega + dt * k3, torque)

        new_omega = omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.satellite.set_angular_velocity(new_omega)