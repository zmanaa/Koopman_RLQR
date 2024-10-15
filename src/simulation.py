import numpy as np


class Simulation:
    def __init__(self, satellite, dynamics, koopman, controller, dt, duration):
        """
        Initializes the Simulation.

        Parameters:
        - satellite (Satellite): Satellite object.
        - dynamics (Dynamics): Dynamics object.
        - koopman (Koopman): Koopman object for data collection.
        - controller (Controller): Controller object.
        - dt (float): Time step.
        - duration (float): Duration of the simulation.
        """
        self.satellite = satellite
        self.dynamics = dynamics
        self.koopman = koopman
        self.controller = controller
        self.dt = dt
        self.duration = duration
        self.time = np.arange(0, duration, dt)
        self.history = []
        self.torque_history = []

    def apply_torque(self, gk):
        """
        Parameters:
        - gk (np.ndarray): Observable vector.

        Returns:
        - torque (np.ndarray): Applied torque vector (3,).
        """
        Kd = self.controller.compute_LQR_gain()
        if Kd is None:
            raise ValueError("Controller gain Kd has not been computed.")
        return -Kd @ gk

    def run(self):
        omegas = []
        torques = []

        for t in self.time:
            current_omega = self.satellite.get_angular_velocity()
            omegas.append(current_omega.copy())

            gk = self.koopman.dictionary_of_observables(current_omega)
            torque = self.apply_torque(gk)
            torques.append(torque.copy())
            self.torque_history.append(torque.copy())

            self.dynamics.step(torque, self.dt)
            self.history.append(current_omega.copy())

        self.koopman.collect_data(omegas, torques)

    def get_simulation_data(self):
        return self.time, np.array(self.history), np.array(self.torque_history)

