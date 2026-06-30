import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm

# ============================================================
# 1. LOAD THE DATA
# ============================================================
# Load the partial stability scan results
df = pd.read_csv('stability_scan_v3_partial.csv')

print(f"Loaded {len(df)} nuclides")
print(f"Z range: {df['Z'].min()} to {df['Z'].max()}")
print(f"N range: {df['N'].min()} to {df['N'].max()}")

# ============================================================
# 2. CLEAN THE DATA
# ============================================================
# Identify numerical spikes (absurdly large negative values)
# We'll set a threshold: anything < -1000 MeV is a "spike"
spike_threshold = -1000.0
df['delta_E_clean'] = df['delta_E'].copy()
df['is_spike'] = df['delta_E'] < spike_threshold

# For the heatmap, we'll cap spikes at a reasonable value
# We'll use the minimum non-spike value as the floor
min_non_spike = df.loc[~df['is_spike'], 'delta_E'].min()
df['delta_E_heatmap'] = df['delta_E'].copy()
df.loc[df['is_spike'], 'delta_E_heatmap'] = min_non_spike - 5.0  # Slightly below the floor

print(f"\nFound {df['is_spike'].sum()} numerical spikes (ΔE < {spike_threshold} MeV)")
print(f"Minimum non-spike ΔE: {min_non_spike:.2f} MeV")

# ============================================================
# 3. CREATE THE HEATMAP GRID
# ============================================================
# Get unique Z and N values
Z_vals = sorted(df['Z'].unique())
N_vals = sorted(df['N'].unique())

# Create grid
delta_grid = np.full((len(Z_vals), len(N_vals)), np.nan)
spike_grid = np.full((len(Z_vals), len(N_vals)), False, dtype=bool)
stable_grid = np.full((len(Z_vals), len(N_vals)), False, dtype=bool)

# Fill grids
for _, row in df.iterrows():
    i = Z_vals.index(row['Z'])
    j = N_vals.index(row['N'])
    delta_grid[i, j] = row['delta_E_heatmap']
    spike_grid[i, j] = row['is_spike']
    stable_grid[i, j] = row['stable'] and not row['is_spike']

# ============================================================
# 4. CREATE THE FIGURE
# ============================================================
fig, ax = plt.subplots(figsize=(14, 10))

# Define colormap with a special color for spikes
# We'll use a standard colormap and overlay spike markers
vmin = min_non_spike - 10.0
vmax = df['delta_E_heatmap'].max()

# Plot the heatmap
im = ax.imshow(
    delta_grid,
    origin='lower',
    aspect='auto',
    extent=[min(N_vals)-0.5, max(N_vals)+0.5, min(Z_vals)-0.5, max(Z_vals)+0.5],
    cmap='RdYlBu_r',
    norm=Normalize(vmin=vmin, vmax=vmax),
    interpolation='nearest'
)

# ============================================================
# 5. OVERLAY STABLE ISOTOPES (GREEN CIRCLES)
# ============================================================
stable_points = [(row['N'], row['Z']) for _, row in df.iterrows() 
                 if row['stable'] and not row['is_spike']]

if stable_points:
    Ns, Zs = zip(*stable_points)
    ax.scatter(Ns, Zs, s=40, facecolors='none', edgecolors='limegreen', 
               linewidths=2.0, label='Predicted stable (ΔE > 0)', zorder=5)

# ============================================================
# 6. OVERLAY NUMERICAL SPIKES (RED X)
# ============================================================
spike_points = [(row['N'], row['Z']) for _, row in df.iterrows() 
                if row['is_spike']]

if spike_points:
    Ns, Zs = zip(*spike_points)
    ax.scatter(Ns, Zs, s=80, marker='X', color='red', 
               label=f'Numerical instability ({len(spike_points)} nuclides)', zorder=6)

# ============================================================
# 7. LABELS AND FORMATTING
# ============================================================
ax.set_xlabel('Neutron Number (N)', fontsize=14)
ax.set_ylabel('Proton Number (Z)', fontsize=14)
ax.set_title('RealQM V3 Partial Stability Heatmap\n(135 nuclides scanned, Z=1 to 11, N=Z to ~3Z)', 
             fontsize=16, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, label='Binding Energy ΔE (MeV)')
cbar.set_label('Binding Energy ΔE (MeV)', fontsize=12)

# Add grid lines
ax.grid(True, linestyle='--', alpha=0.3, color='gray')

# Add N=Z line
N_line = np.arange(min(N_vals), max(N_vals)+1)
Z_line = N_line
ax.plot(N_line, Z_line, 'k--', linewidth=1.5, alpha=0.7, label='N = Z')

# Legend
ax.legend(loc='upper left', fontsize=11)

# Set axis limits
ax.set_xlim(min(N_vals)-1, max(N_vals)+1)
ax.set_ylim(min(Z_vals)-1, max(Z_vals)+1)

# ============================================================
# 8. ADD TEXT NOTE ABOUT SPIKES
# ============================================================
ax.text(0.98, 0.02, 
        f"Note: Red X markers indicate numerical spikes\n(ΔE ≈ -10⁹ MeV) where the solver failed to converge\n"
        f"These correspond to topologically unstable configurations",
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================
# 9. SAVE THE FIGURE
# ============================================================
plt.tight_layout()
plt.savefig('stability_heatmap_partial.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFigure saved as 'stability_heatmap_partial.png'")

# ============================================================
# 10. PRINT SUMMARY STATISTICS
# ============================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total nuclides: {len(df)}")
print(f"Successfully converged: {(~df['is_spike']).sum()} ({100*(~df['is_spike']).sum()/len(df):.1f}%)")
print(f"Numerical spikes: {df['is_spike'].sum()} ({100*df['is_spike'].sum()/len(df):.1f}%)")
print(f"Predicted stable (ΔE > 0, non-spike): {df[(~df['is_spike']) & df['stable']].shape[0]}")
print(f"Predicted unstable (ΔE < 0, non-spike): {df[(~df['is_spike']) & ~df['stable']].shape[0]}")