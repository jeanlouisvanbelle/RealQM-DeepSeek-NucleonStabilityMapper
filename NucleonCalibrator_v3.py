import numpy as np
from scipy.optimize import minimize
import csv
import sys
import traceback
from tqdm import tqdm

print("="*80)
print("RealQM V2.2 - Multi-Nucleus Calibration (^2H, ^4He, ^12C)")
print("="*80)

try:
    # ============================================================
    # 1. PHYSICAL CONSTANTS
    # ============================================================
    e = 1.602176634e-19
    m_p = 1.67262192369e-27
    m_n = 1.67492749804e-27
    c = 299792458.0
    h = 6.62607015e-34
    mu0 = 4e-7 * np.pi
    epsilon0 = 1 / (mu0 * c**2)
    MeV = 1.602176634e-13
    fm = 1e-15

    R_LOOP = 0.8415 * fm
    R_WIRE = 0.1 * fm
    E_SATURATION = 1.0 / R_LOOP**3

    print(f"\n  R_LOOP = {R_LOOP/fm:.4f} fm")
    print(f"  E_SATURATION = {E_SATURATION:.4f} fm⁻³")

    # ============================================================
    # 2. GEOMETRY GENERATOR
    # ============================================================
    def generate_nucleus_geometry(Z, N, seed=42, compaction=0.9):
        A = Z + N
        R0 = compaction * 1.2 * (A ** (1.0/3.0)) * fm
        np.random.seed(seed)
        positions = []
        identities = []
        for i in range(A):
            theta = np.arccos(2 * np.random.rand() - 1)
            phi = 2 * np.pi * np.random.rand()
            r = R0 * np.random.rand() ** (1.0/3.0)
            pos = np.array([r * np.sin(theta) * np.cos(phi),
                            r * np.sin(theta) * np.sin(phi),
                            r * np.cos(theta)])
            positions.append(pos)
            identities.append(0 if i < Z else 1)
        positions = np.array(positions)
        identities = np.array(identities)
        com = np.mean(positions, axis=0)
        positions -= com
        return positions, identities

    # ============================================================
    # 3. FIELD STRENGTH
    # ============================================================
    def field_strength(r, centers, coupling_order=3):
        E_total = 0.0
        for center in centers:
            if np.linalg.norm(r - center) < 1e-30:
                continue
            dist = np.linalg.norm(r - center)
            r_scaled = dist / fm
            field = 1.0 / (r_scaled ** coupling_order + 1e-30)
            E_total += field
        return E_total

    # ============================================================
    # 4. COHERENCE FUNCTION
    # ============================================================
    def coherence_fraction(E, eta_0, E0, Emax=5.0, E_saturation=None):
        if E_saturation is None:
            E_saturation = E_SATURATION
        base = 1 - (1 - eta_0) * np.exp(-E / E0)
        roll_off = np.exp(-(E / Emax) ** 2)
        base = base * roll_off
        saturation = np.exp(-(E / E_saturation) ** 2)
        return base * saturation

    # ============================================================
    # 5. LOOP FUNCTIONS
    # ============================================================
    def generate_loop_points(center, tilt, yaw, radius=R_LOOP, steps=12):
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
        a = 0.1e-15
        L = mu0 * radius * (np.log(8 * radius / a) - 2)
        return L

    def calculate_coulomb_energy(q1, q2, r):
        if r == 0:
            return 1e6
        return (1 / (4 * np.pi * epsilon0)) * q1 * q2 / r

    # ============================================================
    # 6. NUCLEON SOLVER
    # ============================================================
    class NucleonSolverV2_2_Calibrate:
        def __init__(self, positions, identities, parameters):
            self.centers_initial = positions
            self.identities = identities
            self.n_loops = len(positions)
            self.I_p = e * (m_p * c**2 / h)
            self.I_n_base = e * (m_n * c**2 / h)
            self.repulsion_strength = parameters.get('repulsion_strength', 0.95)
            self.alpha_scale = parameters.get('alpha_scale', 0.85)
            self.neutron_eta_0 = 0.676
            self.neutron_E0 = parameters.get('neutron_E0', 0.5)
            self.proton_eta_0 = 1.0
            self.proton_E0 = 1.5
            self.Emax = parameters.get('Emax', 5.0)
            self.E_saturation = E_SATURATION
            self.delta_E = None

        def compute_eta_for_nucleon(self, idx, centers):
            center = centers[idx]
            E = field_strength(center, centers)
            if self.identities[idx] == 0:
                return coherence_fraction(E, self.proton_eta_0, self.proton_E0, self.Emax, self.E_saturation)
            else:
                return coherence_fraction(E, self.neutron_eta_0, self.neutron_E0, self.Emax, self.E_saturation)

        def compute_currents(self, centers):
            currents = []
            for idx in range(self.n_loops):
                eta = self.compute_eta_for_nucleon(idx, centers)
                if self.identities[idx] == 0:
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

        def calculate_energy(self, params):
            centers, angles = self.unpack_params(params)

            loops = []
            for i in range(self.n_loops):
                loops.append(generate_loop_points(centers[i], angles[i, 0], angles[i, 1], steps=12))

            currents = self.compute_currents(centers)
            charges = [e if self.identities[i] == 0 else 0.0 for i in range(self.n_loops)]

            mag_energy = 0.0
            for i in range(self.n_loops):
                L_ii = calculate_self_inductance(loops[i], R_LOOP, charges[i], currents[i])
                mag_energy -= 0.5 * L_ii * currents[i]**2
            for i in range(self.n_loops):
                for j in range(i + 1, self.n_loops):
                    M_ij = calculate_mutual_inductance(loops[i], loops[j])
                    mag_energy += M_ij * currents[i] * currents[j]
            mag_energy_scaled = self.alpha_scale * mag_energy

            coulomb_energy = 0.0
            for i in range(self.n_loops):
                for j in range(i + 1, self.n_loops):
                    if self.identities[i] == 0 and self.identities[j] == 0:
                        r = np.linalg.norm(centers[i] - centers[j])
                        coulomb_energy += calculate_coulomb_energy(e, e, r)

            kinetic_energy = 0.0
            for i in range(self.n_loops):
                if self.identities[i] == 0:
                    kinetic_energy += 0.5 * m_p * c**2
                else:
                    kinetic_energy += 0.5 * m_n * c**2

            core_rep_energy = 0.0
            r0_physical = 2 * R_LOOP
            for i in range(self.n_loops):
                for j in range(i + 1, self.n_loops):
                    r = np.linalg.norm(centers[i] - centers[j])
                    if r > 0:
                        core_rep_energy += self.repulsion_strength * MeV * np.exp(-(r / r0_physical) ** 2)

            total_energy = mag_energy_scaled + coulomb_energy + kinetic_energy + core_rep_energy
            return total_energy / MeV

        def solve(self, verbose=False):
            initial_angles = np.zeros((self.n_loops, 2))
            initial_params = self.pack_params(self.centers_initial, initial_angles)

            half_range = max(5.0, 3.0 * 1.2 * (self.n_loops ** (1.0/3.0)))
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
            initial_params += np.random.uniform(-0.01, 0.01, len(initial_params))

            result = minimize(
                self.calculate_energy,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-8}
            )

            free_energy = 0.0
            for i in range(self.n_loops):
                if self.identities[i] == 0:
                    free_energy += 0.5 * m_p * c**2
                else:
                    free_energy += 0.5 * m_n * c**2
            free_energy = free_energy / MeV

            self.delta_E = free_energy - result.fun
            return result

    # ============================================================
    # 7. MULTI-NUCLEUS CALIBRATION
    # ============================================================
    print("\n--- Multi-nucleus calibration ---")
    print("Sweeping alpha_scale and repulsion_strength...")
    print("Targets: ^2H (2.22 MeV), ^4He (28.30 MeV), ^12C (92.16 MeV)\n")

    targets = {
        (1, 1): 2.22,   # ^2H
        (2, 2): 28.30,  # ^4He
        (6, 6): 92.16,  # ^12C
    }

    best_score = np.inf
    best_params = None

    # Wider sweep to find the right balance
    alpha_range = np.arange(0.70, 0.95, 0.025)
    repulsion_range = np.arange(0.80, 1.40, 0.05)

    results = []
    total = len(alpha_range) * len(repulsion_range)
    count = 0

    for alpha in alpha_range:
        for repulsion in repulsion_range:
            count += 1
            print(f"  Testing {count}/{total}: alpha={alpha:.3f}, repulsion={repulsion:.2f}")

            total_error = 0.0
            nuclide_results = {}

            for (Z, N), target in targets.items():
                positions, identities = generate_nucleus_geometry(Z, N, seed=42, compaction=0.9)
                params = {
                    'alpha_scale': alpha,
                    'repulsion_strength': repulsion,
                    'neutron_E0': 0.5,
                    'Emax': 5.0,
                }

                solver = NucleonSolverV2_2_Calibrate(positions, identities, params)
                solver.solve(verbose=False)

                error = abs(solver.delta_E - target)
                total_error += error
                nuclide_results[f"{Z},{N}"] = {
                    'delta_E': solver.delta_E,
                    'error': error,
                }

            print(f"    D: {nuclide_results['1,1']['delta_E']:.2f} MeV (err={nuclide_results['1,1']['error']:.2f})")
            print(f"    He4: {nuclide_results['2,2']['delta_E']:.2f} MeV (err={nuclide_results['2,2']['error']:.2f})")
            print(f"    C12: {nuclide_results['6,6']['delta_E']:.2f} MeV (err={nuclide_results['6,6']['error']:.2f})")
            print(f"    Total error: {total_error:.2f}")

            results.append({
                'alpha': alpha,
                'repulsion': repulsion,
                'total_error': total_error,
                'delta_D': nuclide_results['1,1']['delta_E'],
                'delta_He4': nuclide_results['2,2']['delta_E'],
                'delta_C12': nuclide_results['6,6']['delta_E'],
                'error_D': nuclide_results['1,1']['error'],
                'error_He4': nuclide_results['2,2']['error'],
                'error_C12': nuclide_results['6,6']['error'],
            })

            if total_error < best_score:
                best_score = total_error
                best_params = params.copy()

    # Sort results
    results.sort(key=lambda x: x['total_error'])

    print("\n" + "="*60)
    print("MULTI-NUCLEUS CALIBRATION RESULTS")
    print("="*60)
    for i, r in enumerate(results[:10]):
        print(f"{i+1}. alpha={r['alpha']:.3f}, repulsion={r['repulsion']:.2f} MeV")
        print(f"   D: {r['delta_D']:.2f} MeV (err={r['error_D']:.2f})")
        print(f"   He4: {r['delta_He4']:.2f} MeV (err={r['error_He4']:.2f})")
        print(f"   C12: {r['delta_C12']:.2f} MeV (err={r['error_C12']:.2f})")
        print(f"   Total error: {r['total_error']:.2f}")

    print(f"\n✅ Best params: alpha={best_params['alpha_scale']:.3f}, repulsion={best_params['repulsion_strength']:.2f} MeV")
    print(f"   Total error: {best_score:.2f} MeV")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()

print("\n" + "="*80)
print("Calibration complete.")
print("="*80)
input("\nPress Enter to close...")