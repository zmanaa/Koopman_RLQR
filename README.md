## Overview

Simulates satellite dynamics using Koopman operators and a Robust LQR Controller. Generates trajectories, computes Koopman operators, and visualizes results through various plots.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/zmanaa/Koopman_RLQR.git
   cd Koopman_RLQR
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main simulation script:

```bash
python experiments/1_satellite_dynamics.py
```

### Outputs

- **Plots**: Saved in the `plots/` directory as PDF files.
- **Logs**: Stored in `logs/simulation.log`.
