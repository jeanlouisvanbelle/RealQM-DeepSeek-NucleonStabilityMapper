import numpy as np
from scipy.optimize import minimize
import csv
import sys
import traceback
from tqdm import tqdm
import os
import time

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
# 2. CALIBRATED PARAMETERS (from multi-nucleus calibration)
# ============================================================
ALPHA_SCALE = 0.850
REPULSION_STRENGTH = 1.30  # MeV
EMAX = 5.0
NEUTRON_E0 = 0.5
COMPACTION = 0.9

# ============================================================
# 3. LOOP GEOMETRY
# ============================================================
R_LOOP = 0.8415 * fm
R_WIRE = 0.1 * fm
E_SATURATION = 1.0 / R_LOOP**3

# ============================================================
# 4. FIELD STRENGTH & COHERENCE
# ============================================================
def field_strength(r, centers, coupling_order=3):
    """Compute field strength at position r."""
    E_total = 0.0
    for center in centers:
        if np.linalg.norm(r - center) < 1e-30:
            continue
        dist = np.linalg.norm(r - center)
        r_scaled = dist / fm
        field = 1.0 / (r_scaled ** coupling_order + 1e-30)
        E_total += field
    return E_total

def coherence_fraction_v2_2(E, eta_0, E0, Emax=EMAX, E_saturation=None):
    """Coherence with field-based saturation (Emax=5.0)."""
    if E_saturation is None:
        E_saturation = E_SATURATION
    base = 1 - (1 - eta_0) * np.exp(-E / E0)
    roll_off = np.exp(-(E / Emax) ** 2)
    base = base * roll_off
    saturation = np.exp(-(E / E_saturation) ** 2)
    return base * saturation

# ============================================================
# 5. INDUCTANCE & ENERGY FUNCTIONS
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

def calculate_volume_exclusion_energy(centers, loops, identities, currents):
    total_penalty = 0.0
    n_loops = len(loops)
    for i in range(n_loops):
        for j in range(i + 1, n_loops):
            center_dist = np.linalg.norm(centers[i] - centers[j])
            if center_dist < 2 * R_LOOP:
                overlap = 1.0 - center_dist / (2 * R_LOOP)
                overlap = max(0.0, min(1.0, overlap))
                h = 2 * R_LOOP - center_dist
                V_overlap = np.pi * h**2 * (3 * R_LOOP - h) / 3
                V_overlap = max(0.0, V_overlap)
                current_product = abs(currents[i] * currents[j]) + 1e-30
                B_field = mu0 * np.sqrt(current_product) / (2 * R_LOOP)
                energy_density = B_field**2 / (2 * mu0)
                penalty = energy_density * V_overlap / MeV
                total_penalty += penalty
    return total_penalty

# ============================================================
# 6. GEOMETRY GENERATOR
# ============================================================
def generate_nucleus_geometry(Z, N, seed=42, compaction=COMPACTION):
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
# 7. NUCLEON SOLVER V2.2 (with calibrated parameters)
# ============================================================
class NucleonSolverV2_2_Scan:
    def __init__(self, positions, identities):
        self.centers_initial = positions
        self.identities = identities
        self.n_loops = len(positions)
        self.n_params = 5 * self.n_loops
        self.Z = np.sum(identities == 0)
        self.N = np.sum(identities == 1)
        self.I_p = e * (m_p * c**2 / h)
        self.I_n_base = e * (m_n * c**2 / h)
        
        # Calibrated parameters
        self.repulsion_strength = REPULSION_STRENGTH
        self.alpha_scale = ALPHA_SCALE
        self.neutron_eta_0 = 0.676
        self.neutron_E0 = NEUTRON_E0
        self.proton_eta_0 = 1.0
        self.proton_E0 = 1.5
        self.Emax = EMAX
        self.E_saturation = E_SATURATION
        
        self.U0 = None
        self.U_min = None
        self.delta_E = None
        self.free_energy = None
        self.final_centers = None
        self.movement_distance = None
        self.mag_attractive_energy = 0.0
        self.repulsive_energy = 0.0
        self.boundary_hit = False
        self.success = False

    def compute_eta_for_nucleon(self, idx, centers):
        center = centers[idx]
        E = field_strength(center, centers)
        if self.identities[idx] == 0:
            return coherence_fraction_v2_2(E, self.proton_eta_0, self.proton_E0, self.Emax, self.E_saturation)
        else:
            return coherence_fraction_v2_2(E, self.neutron_eta_0, self.neutron_E0, self.Emax, self.E_saturation)

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
        
        # Magnetic energy
        mag_energy = 0.0
        for i in range(self.n_loops):
            L_ii = calculate_self_inductance(loops[i], R_LOOP, charges[i], currents[i])
            mag_energy -= 0.5 * L_ii * currents[i]**2
        for i in range(self.n_loops):
            for j in range(i + 1, self.n_loops):
                M_ij = calculate_mutual_inductance(loops[i], loops[j])
                mag_energy += M_ij * currents[i] * currents[j]
        mag_energy_scaled = self.alpha_scale * mag_energy
        
        # Coulomb energy
        coulomb_energy = 0.0
        for i in range(self.n_loops):
            for j in range(i + 1, self.n_loops):
                if self.identities[i] == 0 and self.identities[j] == 0:
                    r = np.linalg.norm(centers[i] - centers[j])
                    coulomb_energy += calculate_coulomb_energy(e, e, r)
        
        # Kinetic energy
        kinetic_energy = 0.0
        for i in range(self.n_loops):
            if self.identities[i] == 0:
                kinetic_energy += 0.5 * m_p * c**2
            else:
                kinetic_energy += 0.5 * m_n * c**2
        
        # Core repulsion
        core_rep_energy = 0.0
        r0_physical = 2 * R_LOOP
        for i in range(self.n_loops):
            for j in range(i + 1, self.n_loops):
                r = np.linalg.norm(centers[i] - centers[j])
                if r > 0:
                    core_rep_energy += self.repulsion_strength * MeV * np.exp(-(r / r0_physical) ** 2)
        
        # Volume exclusion
        volume_exclusion_joules = calculate_volume_exclusion_energy(centers, loops, self.identities, currents)
        
        total_energy = mag_energy_scaled + coulomb_energy + kinetic_energy + core_rep_energy + volume_exclusion_joules
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
        
        try:
            result = minimize(
                self.calculate_energy,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-8}
            )
            
            if not result.success:
                if verbose:
                    print(f"  Warning: Optimiser did not converge (message: {result.message})")
                self.success = False
            else:
                self.success = True
            
            free_energy = 0.0
            for i in range(self.n_loops):
                if self.identities[i] == 0:
                    free_energy += 0.5 * m_p * c**2
                else:
                    free_energy += 0.5 * m_n * c**2
            free_energy = free_energy / MeV
            
            self.U_min = result.fun
            self.free_energy = free_energy
            self.delta_E = free_energy - result.fun
            
            self.final_centers, self.final_angles = self.unpack_params(result.x)
            distances = np.linalg.norm(self.final_centers - self.centers_initial, axis=1)
            self.movement_distance = np.mean(distances) / fm
            
            # Boundary hit detection
            tolerance = 0.01 * half_range
            hit = False
            for i in range(self.n_loops):
                for axis in range(3):
                    lower = self.centers_initial[i, axis]/fm - half_range
                    upper = self.centers_initial[i, axis]/fm + half_range
                    val = result.x[5*i + axis]
                    if (val - lower < tolerance) or (upper - val < tolerance):
                        hit = True
                        break
                if hit:
                    break
            self.boundary_hit = hit
            
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            self.success = False
            self.delta_E = np.nan
            self.movement_distance = np.nan
            self.boundary_hit = False
        
        return self

# ============================================================
# 8. FULL STABILITY SCAN
# ============================================================
def run_full_scan(Z_min=1, Z_max=20, N_ratio_min=1.0, N_ratio_max=3.0,
                  output_file='stability_scan_v2.2_full.csv', verbose=True):
    """
    Full stability scan with instant checkpointing.
    """
    # Count total combinations
    total_combinations = 0
    scan_list = []
    for Z in range(Z_min, Z_max+1):
        for N in range(int(np.ceil(Z*N_ratio_min)), int(np.floor(Z*N_ratio_max))+1):
            total_combinations += 1
            scan_list.append((Z, N))
    
    print("="*80)
    print("RealQM V2.2 Full Stability Scan")
    print("="*80)
    print(f"Calibrated parameters:")
    print(f"  alpha_scale = {ALPHA_SCALE:.3f}")
    print(f"  repulsion_strength = {REPULSION_STRENGTH:.2f} MeV")
    print(f"  Emax = {EMAX:.1f}")
    print(f"  neutron_E0 = {NEUTRON_E0:.1f}")
    print(f"Scanning: Z = {Z_min} to {Z_max}, N = Z to 3Z")
    print(f"Total nuclides: {total_combinations}")
    print("="*80)
    
    # Write header
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Z', 'N', 'A', 'delta_E', 'movement', 'stable', 'boundary_hit', 'success'])
        writer.writeheader()
    
    results = []
    progress = tqdm(total=total_combinations, desc="Scanning nuclides") if verbose else None
    
    start_time = time.time()
    for idx, (Z, N) in enumerate(scan_list, 1):
        if verbose and progress is None:
            print(f"Testing {idx}/{total_combinations}: Z={Z}, N={N} (A={Z+N})")
        
        try:
            positions, identities = generate_nucleus_geometry(Z, N, seed=42+idx, compaction=COMPACTION)
            solver = NucleonSolverV2_2_Scan(positions, identities)
            solver.solve(verbose=False)
            
            delta_E = solver.delta_E
            movement = solver.movement_distance
            stable = delta_E > 0 if not np.isnan(delta_E) else False
            boundary_hit = solver.boundary_hit
            success = solver.success
            
            result = {
                'Z': Z,
                'N': N,
                'A': Z+N,
                'delta_E': delta_E,
                'movement': movement,
                'stable': stable,
                'boundary_hit': boundary_hit,
                'success': success,
            }
            
        except Exception as e:
            if verbose:
                print(f"  Error on Z={Z}, N={N}: {e}")
            result = {
                'Z': Z,
                'N': N,
                'A': Z+N,
                'delta_E': np.nan,
                'movement': np.nan,
                'stable': False,
                'boundary_hit': False,
                'success': False,
            }
        
        results.append(result)
        
        # Instant checkpoint: append to CSV immediately
        with open(output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Z', 'N', 'A', 'delta_E', 'movement', 'stable', 'boundary_hit', 'success'])
            writer.writerow(result)
        
        if progress:
            progress.update(1)
            # Update description with elapsed time
            elapsed = time.time() - start_time
            if idx % 10 == 0:
                progress.set_description(f"Scanning nuclides ({idx}/{total_combinations}, {elapsed/60:.1f} min)")
    
    if progress:
        progress.close()
    
    print("\n" + "="*80)
    print("Scan complete!")
    print(f"  Total nuclides: {len(results)}")
    print(f"  Results saved to: {output_file}")
    print("="*80)
    
    return results

# ============================================================
# 9. ANALYSE RESULTS
# ============================================================
def analyse_results(results):
    """Print summary statistics."""
    total = len(results)
    success_count = sum(1 for r in results if r['success'])
    stable_count = sum(1 for r in results if r['stable'])
    boundary_hit_count = sum(1 for r in results if r['boundary_hit'])
    nan_count = sum(1 for r in results if np.isnan(r['delta_E']))
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"  Total nuclides:        {total}")
    print(f"  Successful optimisations: {success_count} ({100*success_count/total:.1f}%)")
    print(f"  Predicted stable:      {stable_count} ({100*stable_count/total:.1f}%)")
    print(f"  Boundary hits:         {boundary_hit_count} ({100*boundary_hit_count/total:.1f}%)")
    print(f"  Failed/NaN:            {nan_count} ({100*nan_count/total:.1f}%)")
    
    # Top 10 most bound nuclides
    valid = [r for r in results if not np.isnan(r['delta_E'])]
    valid.sort(key=lambda x: x['delta_E'], reverse=True)
    
    print("\n  Top 10 most bound nuclides:")
    for i, r in enumerate(valid[:10]):
        print(f"    {i+1}. Z={r['Z']:2d}, N={r['N']:2d}, A={r['A']:2d}: ΔE={r['delta_E']:.2f} MeV")

# ============================================================
# 10. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("="*80)
    print("RealQM V2.2 Full Stability Scan")
    print("="*80)
    print("Saturation mechanisms:")
    print("  1. Geometric Loop Volume Exclusion")
    print("  2. Field-Based Coherence Saturation")
    print("Instant checkpointing: Enabled (saves every nuclide)")
    print("Calibrated parameters (from multi-nucleus calibration):")
    print(f"  alpha_scale = {ALPHA_SCALE:.3f}")
    print(f"  repulsion_strength = {REPULSION_STRENGTH:.2f} MeV")
    print(f"  Emax = {EMAX:.1f}")
    print("="*80)
    
    # Run the scan
    results = run_full_scan(
        Z_min=1,
        Z_max=20,
        N_ratio_min=1.0,
        N_ratio_max=3.0,
        output_file='stability_scan_v2.2_full.csv',
        verbose=True
    )
    
    # Analyse results
    analyse_results(results)
    
    print("\n" + "="*80)
    print("Scan complete. You can stop the script now.")
    print("The CSV file contains all results.")
    print("="*80)
    
    input("\nPress Enter to close...")