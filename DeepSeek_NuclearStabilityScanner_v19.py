import numpy as np
from scipy.optimize import minimize
import time
import csv
import sys
import traceback

# ============================================================
# 1. PHYSICAL CONSTANTS
# ============================================================
e = 1.602176634e-19
m_p = 1.67262192369e-27
m_n = 1.67492749804e-27
c = 299792458.0
h = 6.62607015e-34
hbar = h / (2 * np.pi)
mu0 = 4e-7 * np.pi
epsilon0 = 1 / (mu0 * c**2)
MeV = 1.602176634e-13
fm = 1e-15

# ============================================================
# 2. NEUTRON GEOMETRY
# ============================================================
R_plus = 0.8415 * fm
R_minus = 0.478 * fm
r0_physical = 2 * R_plus

print("=" * 80)
print("RealQM Calibration V19: Full Sweep")
print("=" * 80)
print("Sweeping alpha_scale (0.8–1.2) and repulsion (0.25–0.75)")
print("Target: ^4He = 28.296 MeV")
print("Optimizer is alive and taking steps.")
print("=" * 80)

# ============================================================
# 3. LOCAL FIELD STRENGTH
# ============================================================
def local_field_strength(r, coupling_order=3):
    if r == 0:
        return 1e6
    r_scaled = r / fm
    return 1.0 / (r_scaled ** coupling_order + 1e-30)

def coherence_fraction(E, eta_0, E0, Emax=None):
    if Emax is None or Emax == np.inf:
        return 1 - (1 - eta_0) * np.exp(-E / E0)
    else:
        base = 1 - (1 - eta_0) * np.exp(-E / E0)
        roll_off = np.exp(-(E / Emax) ** 2)
        return base * roll_off

# ============================================================
# 4. NEUMANN INDUCTANCE ENGINE
# ============================================================
def generate_loop_points(center, tilt, yaw, radius=0.8415e-15, steps=12):
    t = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    base_points = np.zeros((steps, 3))
    base_points[:, 0] = radius * np.cos(t)
    base_points[:, 1] = radius * np.sin(t)
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_tilt = np.array([[1.0, 0.0, 0.0], [0.0, cos_t, -sin_t], [0.0, sin_t, cos_t]])
    R_yaw = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    return np.dot(base_points, np.dot(R_yaw, R_tilt).T) + center

def calculate_mutual_inductance(loop1, loop2):
    n = len(loop1)
    dl1 = np.zeros((n, 3))
    dl1[:-1] = loop1[1:] - loop1[:-1]
    dl1[-1] = loop1[0] - loop1[-1]
    dl2 = np.zeros((n, 3))
    dl2[:-1] = loop2[1:] - loop2[:-1]
    dl2[-1] = loop2[0] - loop2[-1]
    diff = loop1[:, np.newaxis, :] - loop2[np.newaxis, :, :]
    r12 = np.linalg.norm(diff, axis=2)
    core_buffer = 0.1e-15
    r12 = np.where(r12 < core_buffer, core_buffer, r12)
    M = -(mu0 / (4 * np.pi)) * np.sum(np.sum(dl1[:, np.newaxis, :] * dl2[np.newaxis, :, :], axis=2) / r12)
    return M

def calculate_self_inductance(loop, radius, q, I):
    a_wire = 0.1e-15
    L = mu0 * radius * (np.log(8 * radius / a_wire) - 2)
    return L

def calculate_coulomb_energy(q1, q2, r):
    if r == 0:
        return 1e6
    return (1 / (4 * np.pi * epsilon0)) * q1 * q2 / r

# ============================================================
# 5. HELIUM GEOMETRY GENERATOR
# ============================================================
def generate_helium_geometry(Z=2, N=2):
    radius_scale = 0.2 * fm
    positions = []
    for i in range(Z):
        angle = 2 * np.pi * i / max(Z, 1) + 0.1 * (np.random.rand() - 0.5)
        r = radius_scale * (1 + 0.1 * (np.random.rand() - 0.5))
        positions.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    for i in range(N):
        angle = 2 * np.pi * i / max(N, 1) + 0.1 * (np.random.rand() - 0.5) + np.pi / N
        r = radius_scale * (1 + 0.1 * (np.random.rand() - 0.5))
        positions.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    positions = np.array(positions)
    identities = np.array([0] * Z + [1] * N)
    return positions, identities

# ============================================================
# 6. NUCLEON SOLVER V19
# ============================================================
class NucleonSolverV19:
    def __init__(self, positions, identities, parameters):
        self.centers_initial = positions
        self.identities = identities
        self.n_loops = len(positions)
        self.n_params = 5 * self.n_loops
        self.Z = np.sum(identities == 0)
        self.N = np.sum(identities == 1)
        self.I_p = e * (m_p * c**2 / h)
        self.I_n_base = e * (m_n * c**2 / h)
        self.repulsion_strength = parameters['repulsion_strength']
        self.neutron_eta_0 = 0.676
        self.neutron_E0 = parameters['neutron_E0']
        self.proton_eta_0 = 1.0
        self.proton_E0 = 1.5
        self.Emax = parameters['Emax']
        self.alpha_scale = parameters.get('alpha_scale', 1.0)
        self.r0 = r0_physical
        
        self.U0 = None
        self.U_min = None
        self.delta_E = None
        self.free_energy = None
        self.final_centers = None
        self.movement_distance = None
        self.mag_attractive_energy = None
        self.repulsive_energy = None

    def compute_eta_for_nucleon(self, idx, centers):
        center = centers[idx]
        E_total = 0.0
        for j, other in enumerate(centers):
            if j == idx:
                continue
            r = np.linalg.norm(center - other)
            E_total += local_field_strength(r, coupling_order=3)
        if self.identities[idx] == 0:
            return coherence_fraction(E_total, self.proton_eta_0, self.proton_E0, self.Emax)
        else:
            return coherence_fraction(E_total, self.neutron_eta_0, self.neutron_E0, self.Emax)

    def compute_currents(self, centers):
        currents = []
        for idx, identity in enumerate(self.identities):
            eta = self.compute_eta_for_nucleon(idx, centers)
            if identity == 0:
                currents.append(self.I_p * eta)
            else:
                currents.append(self.I_n_base * eta)
        return currents

    def pack_params(self, centers, angles):
        params = np.zeros(5 * self.n_loops)
        for i in range(self.n_loops):
            params[5*i : 5*i + 3] = centers[i] / fm
            params[5*i + 3 : 5*i + 5] = angles[i]
        return params

    def unpack_params(self, params):
        centers = np.zeros((self.n_loops, 3))
        angles = np.zeros((self.n_loops, 2))
        for i in range(self.n_loops):
            centers[i] = params[5*i : 5*i + 3] * fm
            angles[i] = params[5*i + 3 : 5*i + 5]
        return centers, angles

    def calculate_energy_components(self, params):
        centers, angles = self.unpack_params(params)
        loops = []
        for i in range(self.n_loops):
            loops.append(generate_loop_points(centers[i], angles[i, 0], angles[i, 1]))
        currents = self.compute_currents(centers)
        charges = [e if self.identities[i] == 0 else 0.0 for i in range(self.n_loops)]
        
        total_energy_joules = 0.0
        mag_attractive_joules = 0.0
        repulsive_joules = 0.0
        radius = 0.8415e-15

        # 1. Magnetic energy (attractive)
        mag_joules = 0.0
        for i in range(self.n_loops):
            L_ii = calculate_self_inductance(loops[i], radius, charges[i], currents[i])
            mag_joules -= 0.5 * L_ii * currents[i]**2
        for i in range(self.n_loops):
            for j in range(i + 1, self.n_loops):
                M_ij = calculate_mutual_inductance(loops[i], loops[j])
                mag_joules += M_ij * currents[i] * currents[j]
        mag_joules_scaled = self.alpha_scale * mag_joules
        mag_attractive_joules += mag_joules_scaled

        # 2. Coulomb repulsion
        coulomb_joules = 0.0
        for i in range(self.n_loops):
            for j in range(i + 1, self.n_loops):
                if self.identities[i] == 0 and self.identities[j] == 0:
                    r = np.linalg.norm(centers[i] - centers[j])
                    coulomb_joules += calculate_coulomb_energy(e, e, r)
        repulsive_joules += coulomb_joules

        # 3. Kinetic energy
        kinetic_joules = 0.0
        for i in range(self.n_loops):
            if self.identities[i] == 0:
                kinetic_joules += 0.5 * m_p * c**2
            else:
                kinetic_joules += 0.5 * m_n * c**2

        # 4. Gaussian repulsion
        core_rep_joules = 0.0
        for i in range(self.n_loops):
            for j in range(i + 1, self.n_loops):
                r = np.linalg.norm(centers[i] - centers[j])
                if r > 0:
                    core_rep_joules += self.repulsion_strength * MeV * np.exp(-(r / self.r0) ** 2)
        repulsive_joules += core_rep_joules

        total_energy_joules = mag_joules_scaled + coulomb_joules + kinetic_joules + core_rep_joules

        return (total_energy_joules / MeV,
                mag_attractive_joules / MeV,
                repulsive_joules / MeV)

    def calculate_energy(self, params):
        total_mev, _, _ = self.calculate_energy_components(params)
        return total_mev

    def calculate_free_nucleon_energy(self):
        total_energy_joules = 0.0
        for i in range(self.n_loops):
            if self.identities[i] == 0:
                total_energy_joules += 0.5 * m_p * c**2
            else:
                total_energy_joules += 0.5 * m_n * c**2
        return total_energy_joules / MeV

    def solve(self, verbose=False):
        initial_angles = np.zeros((self.n_loops, 2))
        initial_params = self.pack_params(self.centers_initial, initial_angles)
        
        self.U0 = self.calculate_energy(initial_params)

        half_range = 2.0
        angle_range = np.pi / 2
        bounds = []
        for i in range(self.n_loops):
            bounds.extend([
                (self.centers_initial[i, 0]/fm - half_range, self.centers_initial[i, 0]/fm + half_range),
                (self.centers_initial[i, 1]/fm - half_range, self.centers_initial[i, 1]/fm + half_range),
                (self.centers_initial[i, 2]/fm - half_range, self.centers_initial[i, 2]/fm + half_range)
            ])
            bounds.extend([
                (-angle_range, angle_range),
                (-angle_range, angle_range)
            ])

        np.random.seed(42)
        initial_params += np.random.uniform(-0.01, 0.01, self.n_params)

        result = minimize(
            self.calculate_energy,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        self.U_min = result.fun
        self.free_energy = self.calculate_free_nucleon_energy()
        self.delta_E = self.free_energy - self.U_min
        
        self.final_centers, self.final_angles = self.unpack_params(result.x)
        distances = np.linalg.norm(self.final_centers - self.centers_initial, axis=1)
        self.movement_distance = np.mean(distances) / fm
        
        _, mag_att, rep = self.calculate_energy_components(result.x)
        self.mag_attractive_energy = mag_att
        self.repulsive_energy = rep
        
        if verbose:
            print(f"    Iterations: {result.nit}")
            print(f"    ΔE: {self.delta_E:.4f} MeV")
            print(f"    Movement: {self.movement_distance:.3f} fm")
        
        return result

# ============================================================
# 7. SCORING FUNCTION
# ============================================================
def score_helium(delta_E, target=28.296):
    return -abs(delta_E - target)

# ============================================================
# 8. MAIN CALIBRATION SWEEP
# ============================================================
if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("RealQM Calibration V19: Full Sweep")
        print("=" * 80)
        print("Sweeping alpha_scale (0.8–1.2) and repulsion (0.25–0.75)")
        print("Target: ^4He = 28.296 MeV")
        print("Optimizer is alive and taking steps.")
        print("=" * 80)
        
        Emax = 0.5
        neutron_E0 = 0.5
        
        alpha_scales = np.arange(0.8, 1.25, 0.05)
        repulsions = np.arange(0.25, 0.80, 0.05)
        
        results = []
        total = len(alpha_scales) * len(repulsions)
        count = 0
        
        for alpha_scale in alpha_scales:
            for repulsion in repulsions:
                count += 1
                print(f"\nTesting {count}/{total}: alpha_scale={alpha_scale:.2f}, repulsion={repulsion:.2f}")
                
                parameters = {
                    'alpha_scale': alpha_scale,
                    'repulsion_strength': repulsion,
                    'neutron_E0': neutron_E0,
                    'Emax': Emax,
                }
                
                try:
                    positions, identities = generate_helium_geometry(Z=2, N=2)
                    solver = NucleonSolverV19(positions, identities, parameters)
                    solver.solve(verbose=False)
                    
                    score = score_helium(solver.delta_E)
                    
                    results.append({
                        'alpha_scale': alpha_scale,
                        'repulsion': repulsion,
                        'delta_E': solver.delta_E,
                        'score': score,
                        'movement': solver.movement_distance,
                    })
                    
                    print(f"  ΔE: {solver.delta_E:.4f} MeV | Score: {score:.4f}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    results.append({
                        'alpha_scale': alpha_scale,
                        'repulsion': repulsion,
                        'delta_E': np.nan,
                        'score': -np.inf,
                        'movement': 0,
                    })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print("\n" + "=" * 80)
        print("RANKED RESULTS (Top 20)")
        print("=" * 80)
        print(f"{'Rank':<5} | {'alpha':<8} | {'repulsion':<10} | {'ΔE (MeV)':<12} | {'Diff (MeV)':<12} | {'Movement (fm)':<12}")
        print("-" * 80)
        for i, result in enumerate(results[:20]):
            diff = abs(result['delta_E'] - 28.296) if not np.isnan(result['delta_E']) else np.nan
            print(f"{i+1:<5} | {result['alpha_scale']:<8.2f} | {result['repulsion']:<10.2f} | {result['delta_E']:<12.4f} | {diff:<12.4f} | {result['movement']:<12.3f}")
        
        print("\n" + "=" * 80)
        print("BEST PARAMETER SET")
        print("=" * 80)
        best = results[0]
        print(f"alpha_scale: {best['alpha_scale']:.2f}")
        print(f"repulsion_strength: {best['repulsion']:.2f} MeV")
        print(f"ΔE: {best['delta_E']:.4f} MeV (target: 28.296 MeV)")
        print(f"Difference: {abs(best['delta_E'] - 28.296):.4f} MeV")
        print(f"Movement: {best['movement']:.3f} fm")
        print("=" * 80)
        
        try:
            with open('calibration_v19_full_results.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Rank', 'alpha_scale', 'repulsion', 'delta_E', 'score', 'movement'])
                for i, result in enumerate(results):
                    writer.writerow([i+1, result['alpha_scale'], result['repulsion'], result['delta_E'], result['score'], result['movement']])
            print("\n✓ Results saved to 'calibration_v19_full_results.csv'")
        except PermissionError:
            print("\n⚠️ PermissionError: Could not save CSV.")
        
        print("\n" + "=" * 80)
        print("CALIBRATION V19 COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED:")
        print("=" * 80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        print("=" * 80)
    
    input("\nPress Enter to exit...")