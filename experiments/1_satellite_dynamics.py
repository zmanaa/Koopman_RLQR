from utils.utils import *


plt.style.use('../utils/my_sty.mplstyle')

setup_logging()
np.random.seed(7899)

J = np.array([[40, 1.2, 0.9], [1.2, 17, 1.4], [0.9, 1.4, 15]])
num_trajectories = 100
duration = 5.0
dt = 0.01

Bd1 = np.zeros((9, 9))

koopman = Koopman()

simulation = run_simulation(num_trajectories, duration, dt, J, koopman)

t, y, torque = simulation.get_simulation_data()
plot_simulation_data(t, y, torque)

Alift, Blift = koopman.compute_koopman_operators()
logging.info(f"Koopman operators computed with shape X: {np.array(koopman.X).shape}")

plot_koopman_operators(Alift)
logging.info("Koopman operators plotted successfully.")

controller = RobustLQRController(
    Alift=Alift, Blift=Blift, Clift=np.eye(9), Bd1=Bd1
)

Kd = controller.compute_LQR_gain()
logging.info("Optimal LQR gain Kd computed successfully.")

plot_lqr_gain(Kd)

plot_eigenvalues(Alift, Blift, Kd)
logging.info("Eigenvalues plotted successfully.")

