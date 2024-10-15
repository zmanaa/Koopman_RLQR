import numpy as np
import cvxpy as cp


class RobustLQRController:
    def __init__(self, Alift, Blift, Clift, Bd1):
        """
        Initializes the Robust LQR Controller based on Theorem 1.

        Parameters:
        - Alift (np.ndarray): Lifted state matrix (9 x 9).
        - Blift (np.ndarray): Lifted input matrix (9 x 3).
        - Clift (np.ndarray): Pulling the states back matrix.
        - Bd1 (np.ndarray): Disturbance input matrix (3 x 3).
        """
        self.Alift = Alift
        self.Blift = Blift
        self.C = Clift
        self.Bd1 = Bd1
        self.Kd = None  # To store the computed gain

    def compute_LQR_gain(self):
        """
        Computes the robust LQR gain Kd by solving the LMI optimization problem.

        Returns:
        - Kd (np.ndarray): Optimal gain matrix
        """
        n_x = self.Alift.shape[0]  # 9
        n_u = self.Blift.shape[1]  # 3

        Pd = cp.Variable((n_x, n_x), symmetric=True)
        Fd = cp.Variable((n_u, n_x))
        gamma = cp.Variable(1)
        I = np.eye(n_x)

        top_row = [
            Pd,
            self.Alift @ Pd - self.Blift @ Fd,
            self.Bd1,
            np.zeros((n_x, n_x)),
        ]
        second_row = [
            (self.Alift @ Pd - self.Blift @ Fd).T,
            Pd,
            np.zeros((n_x, n_x)),
            Pd.T @ self.C.T,
        ]
        third_row = [
            self.Bd1.T,
            np.zeros((n_x, n_x)),
            gamma * I,
            np.zeros((n_x, n_x)),
        ]
        fourth_row = [
            np.zeros((n_x, n_x)),
            (Pd.T @ self.C.T).T,
            np.zeros((n_x, n_x)),
            gamma * I,
        ]

        LMI = cp.bmat([top_row, second_row, third_row, fourth_row])

        constraints = [
            Pd >> 1e-6 * np.eye(n_x),
            LMI << 0,
            gamma >= 1e-6
        ]

        objective = cp.Minimize(gamma)

        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.SCS, eps=1e-4)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print("Problem status:", problem.status)
            if problem.status == "infeasible":
                print("Infeasibility detected. Inspecting constraints...")
                for constraint in constraints:
                    print(constraint.dual_variables)
            return None

        P_opt = Pd.value
        F_opt = Fd.value

        try:
            P_inv = np.linalg.inv(P_opt)
            self.Kd = F_opt @ P_inv
        except np.linalg.LinAlgError:
            print("P matrix is singular and cannot be inverted.")
            return None

        return self.Kd

    def get_gain(self):
        """
        Returns the computed gain matrix Kd.

        Returns:
        - Kd (np.ndarray): Optimal gain matrix
        """
        if self.Kd is None:
            self.compute_LQR_gain()
        return self.Kd


class DataCollectionController:
    def __init__(self):
        self.Kd = None

    @staticmethod
    def compute_LQR_gain():
        return np.random.random((3, 9))

    def get_gain(self):
        """
        Returns the computed gain matrix Kd.

        Returns:
        - Kd (np.ndarray): Optimal gain matrix
        """
        if self.Kd is None:
            self.compute_LQR_gain()
        return self.Kd
