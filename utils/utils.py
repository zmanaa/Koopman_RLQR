import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import logging
from src.dynamics import Dynamics, Satellite
from src.model import Koopman
from src.controller import RobustLQRController, DataCollectionController
from src.simulation import Simulation

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("../logs/simulation.log"),
            logging.StreamHandler()
        ]
    )


def run_simulation(num_trajectories, duration, dt, J, koopman):
    Bd1 = np.zeros((9, 9))
    for traj in range(num_trajectories):
        initial_omega = np.random.uniform(-0.2 * np.pi, 0.2 * np.pi, size=3)
        satellite = Satellite(inertia_matrix=J, initial_omega=initial_omega)
        dynamics = Dynamics(satellite=satellite)
        controller = DataCollectionController()
        simulation = Simulation(
            satellite=satellite,
            dynamics=dynamics,
            koopman=koopman,
            controller=controller,
            dt=dt,
            duration=duration,
        )
        simulation.run()
        logging.info(f"Completed trajectory {traj + 1}/{num_trajectories}")
    return simulation


def plot_simulation_data(t, y, torque):
    plt.figure()
    plt.plot(t, y[:, 0], label=r'$\omega_x$')
    plt.plot(t, y[:, 1], label=r'$\omega_y$')
    plt.plot(t, y[:, 2], label=r'$\omega_z$')
    plt.legend()
    plt.savefig('../plots/sample_trajectory.pdf', transparent=False, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(t, torque[:, 0], alpha=0.5, label=r'$\tau_x$')
    plt.plot(t, torque[:, 1], alpha=0.5, label=r'$\tau_y$')
    plt.plot(t, torque[:, 2], alpha=0.5, label=r'$\tau_z$')
    plt.legend()
    plt.savefig('../plots/simulation_torque.pdf', transparent=False, bbox_inches='tight')
    plt.close()


def plot_koopman_operators(Alift):
    plt.figure()
    plt.imshow(Alift)
    plt.colorbar(extend="both")
    plt.axis("off")
    plt.savefig('../plots/koopman_operators.pdf', transparent=False, bbox_inches='tight')
    plt.close()


def plot_lqr_gain(Kd):
    plt.figure()
    plt.imshow(Kd)
    plt.colorbar(extend="both")
    plt.axis("off")
    plt.savefig('../plots/lqr_gain.pdf', transparent=False, bbox_inches='tight')
    plt.close()


def plot_eigenvalues(Alift, Blift, Kd):
    eigenvalues_cl, _ = np.linalg.eig(Alift - Blift @ Kd)
    eigenvalues, _ = np.linalg.eig(Alift)

    theta = np.linspace(0, 2 * np.pi, 5000)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    plt.figure()
    plt.plot(x_circle, y_circle, "k--", label="Unit Circle")
    plt.scatter(np.real(eigenvalues_cl), np.imag(eigenvalues_cl), label=r"$\lambda_i$ stable")
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), label=r"$\lambda_i$ unstable")
    plt.xlim([0.9995, 1.0005])
    plt.ylim([-0.07, 0.07])
    plt.legend()
    plt.savefig('../plots/eigenvalues.pdf', transparent=False, bbox_inches='tight')
    plt.close()
