import scipy.linalg as la
import numpy as np


class Koopman:
    def __init__(self):
        self.X = []
        self.Y = []
        self.U = []

    def collect_data(self, omegas, torques):
        """
        Collects data to build the X, Y, and U matrices.
        omegas: List of angular velocity vectors
        torques: List of torque vectors
        """
        self.X.append(np.array(omegas[:-1]))
        self.Y.append(np.array(omegas[1:]))
        self.U.append(np.array(torques[:-1]))

    def get_matrices(self):
        """
        Concatenates all collected data into global X, Y, U matrices.

        Returns:
        - X (np.ndarray): Concatenated state matrix.
        - Y (np.ndarray): Concatenated next state matrix.
        - U (np.ndarray): Concatenated input matrix.
        """
        X = np.vstack(self.X).T
        Y = np.vstack(self.Y).T
        U = np.vstack(self.U).T
        return X, Y, U

    @staticmethod
    def dictionary_of_observables(omega):
        """
        Parameters:
        - omega (np.ndarray): Angular velocity vector (3,).

        Returns:
        - g (np.ndarray): Observable vector.
        """
        omega_x, omega_y, omega_z = omega
        return np.array(
            [
                omega_x,
                omega_y,
                omega_z,
                omega_x**2,
                omega_x * omega_y,
                omega_x * omega_z,
                omega_y**2,
                omega_y * omega_z,
                omega_z**2,
            ]
        )

    def compute_lifted_matrices(self):
        """
        Compute the lifted data sets X_lift and Y_lift based on observables.

        Returns:
        - X_lift (np.ndarray): Lifted X matrix.
        - Y_lift (np.ndarray): Lifted Y matrix.
        """
        X, Y, U = self.get_matrices()
        X_lift = np.array([self.dictionary_of_observables(omega)
                          for omega in X.T]).T
        Y_lift = np.array([self.dictionary_of_observables(omega)
                          for omega in Y.T]).T
        return X_lift, Y_lift

    def compute_koopman_operators(self):
        """
        Solve for the Koopman operators Alift and Blift using the EDMD approach.

        Returns:
        - Alift (np.ndarray): Lifted state matrix.
        - Blift (np.ndarray): Lifted input matrix.
        """
        X_lift, Y_lift = self.compute_lifted_matrices()
        X, Y, U = self.get_matrices()

        combined_matrix = np.vstack((X_lift, U))

        Alift_Blift = Y_lift @ la.pinv(combined_matrix)

        n_x_lift = X_lift.shape[0]
        n_u = U.shape[0]
        Alift = Alift_Blift[:, :n_x_lift]
        Blift = Alift_Blift[:, n_x_lift:]

        return Alift, Blift
