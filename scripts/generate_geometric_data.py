import os
import math
import argparse
import time
import tracemalloc
import numpy as np
import h5py
from cytools import fetch_polytopes, Polytope
from concurrent.futures import ProcessPoolExecutor
from geometry_charge_conventions import canonicalize_unique_charge_rows

def generate_and_save_geometry(h11, cy, poly_points, simplices, filepath):
    # --- 1. Basic Geometric Quantities ---
    h21 = int(cy.h21())
    glsm = np.array(cy.glsm_charge_matrix(include_origin=False), dtype=int)
    basis = np.array(cy.divisor_basis(), dtype=int)
    
    # Find tip of SKC
    n_val, m_val = 1.0, 1.0
    tip = np.array(cy.toric_kahler_cone().tip_of_stretched_cone(math.sqrt(n_val)), dtype=float)
    
    qprime_raw = np.array(cy.toric_effective_cone().rays(), dtype=float)
    qprime, charge_metadata = canonicalize_unique_charge_rows(qprime_raw)
    nq = qprime.shape[0]
    
    # PTD volumes at tip
    div_vols = np.array(cy.compute_divisor_volumes(tip))
    python_basis = basis - 1  
    tau0 = div_vols[python_basis]
    
    # Compute inverse Kahler metric (Assuming cytools >= 0.8.0)
    Kinv0_raw = np.array(cy.compute_inverse_kahler_metric(tip))
    Kinv0 = 0.5 * (Kinv0_raw + Kinv0_raw.T) # Hermitian / symmetric part
    
    tau = np.copy(tau0)
    Kinv = np.copy(Kinv0)
    
    # --- 2. Iterate to satisfy constraint (m loop) ---
    rhs_constraint = np.zeros(nq)
    lhs_constraint = np.zeros((nq, nq))
    
    while True:
        lhs_constraint.fill(0.0)
        for j in range(nq):
            qj = qprime[j, :]
            Kinv_qj = Kinv @ qj
            tau_qj = np.dot(tau, qj)
            
            for i in range(j + 1, nq):
                qi = qprime[i, :]
                lhs_constraint[i, j] = abs(
                    math.log(abs(math.pi * np.dot(qi, Kinv_qj))) 
                    - 2 * math.pi * np.dot(tau, qi + qj)
                )
            rhs_constraint[j] = abs(math.log(abs(tau_qj)) - 2 * math.pi * tau_qj)
            
        converged = True
        for i in range(nq):
            for j in range(i):
                if lhs_constraint[i, j] <= rhs_constraint[i]:
                    converged = False
                    break
            if not converged: break
            
        if converged:
            break
            
        m_val += 1e-2
        m2 = m_val**2
        m4 = m2**2
        tau = m2 * tau0
        Kinv = m4 * Kinv0

    # Adjust tip if tau < 1
    if np.min(tau) <= 1.0:
        n_val = 1.0 / np.min(tau)
        tip = math.sqrt(n_val) * tip
        div_vols = np.array(cy.compute_divisor_volumes(tip))
        tau = div_vols[python_basis]
        Kinv_raw = np.array(cy.compute_inverse_kahler_metric(tip))
        Kinv = 0.5 * (Kinv_raw + Kinv_raw.T)
        
    tip_prefactor = np.array([math.sqrt(n_val), m_val], dtype=float)
    V = float(cy.compute_cy_volume(tip))
    
    # --- 3. Instanton charges (Q) and Amplitudes (L) ---
    num_cross = (nq * (nq - 1)) // 2
    term_count = nq + num_cross
    q = np.zeros((int(cy.h11()), term_count), dtype=float)
    L_raw = np.zeros((2, term_count), dtype=float)
    
    q[:, 0:nq] = qprime.T

    idx = 0
    for i in range(nq - 1):
        for j in range(i + 1, nq):
            q[:, nq + idx] = qprime[j, :] - qprime[i, :]
            
            term1 = (math.pi * np.dot(qprime[i, :], Kinv @ qprime[j, :]) +
                     np.dot((qprime[i, :] + qprime[j, :]), tau)) * 8 * math.pi / (V**2)
            term2 = -2 * math.log10(math.e) * math.pi * (np.dot(qprime[i, :], tau) + np.dot(qprime[j, :], tau))
            
            L_raw[0, nq + idx] = term1
            L_raw[1, nq + idx] = term2
            idx += 1

    for j in range(nq):
        L_raw[0, j] = (8 * math.pi / (V**2)) * np.dot(qprime[j, :], tau)
        L_raw[1, j] = -2 * math.log10(math.e) * math.pi * np.dot(qprime[j, :], tau)

    L = np.zeros_like(L_raw)
    # L[:, 0] = sign of term1; L[:, 1] = log10(|term1|) + term2
    L[0, :] = np.sign(L_raw[0, :])
    L[1, :] = np.log10(np.abs(L_raw[0, :])) + L_raw[1, :]

    # --- 4. Write exactly matching HDF5 ---
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, "w") as file:
        f1 = file.create_group("cytools")
        
        # cytools/geometric
        f1a = f1.create_group("geometric")
        f1a.create_dataset("points", data=poly_points, compression="gzip", compression_opts=9)
        f1a.create_dataset("simplices", data=simplices, compression="gzip", compression_opts=9)
        f1a.create_dataset("h21", data=h21)
        f1a.create_dataset("glsm", data=glsm, compression="gzip", compression_opts=9)
        f1a.create_dataset("basis", data=basis, compression="gzip", compression_opts=9)
        f1a.create_dataset("tip", data=tip, compression="gzip", compression_opts=9)
        f1a.create_dataset("tip_prefactor", data=tip_prefactor, compression="gzip", compression_opts=9)
        f1a.create_dataset("CY_volume", data=V)
        f1a.create_dataset("divisor_volumes", data=tau, compression="gzip", compression_opts=9)
        f1a.create_dataset("Kinv", data=Kinv, compression="gzip", compression_opts=9)
        f1a.create_dataset("effective_cone", data=qprime, compression="gzip", compression_opts=9)
        f1a.attrs["potential_charge_convention"] = charge_metadata["convention"]
        f1a.attrs["raw_effective_cone_ray_count"] = charge_metadata["raw_count"]
        f1a.attrs["canonical_effective_cone_ray_count"] = charge_metadata["canonical_count"]
        f1a.attrs["duplicate_effective_cone_rows_removed"] = charge_metadata[
            "duplicates_removed"
        ]
        
        # cytools/potential
        f1b = f1.create_group("potential")
        f1b.create_dataset("L", data=L, compression="gzip", compression_opts=9)
        f1b.create_dataset("Q", data=q, compression="gzip", compression_opts=9)

def process_single_polytope(args):
    """Worker function executed on isolated CPU cores."""
    tri_idx, points, h11, base_dir = args
    try:
        # Reconstruct the Polytope from raw points to avoid pickling errors
        poly = Polytope(points)
        t = poly.triangulate()
        simplices = np.array(t.simplices(), dtype=int)
        cy = t.get_cy()
        
        dir_h11 = f"h11_{h11:03d}"
        dir_np = f"np_{tri_idx+1:07d}"
        dir_cy = "cy_0000001"
        filepath = os.path.join(base_dir, dir_h11, dir_np, dir_cy, "cyax.h5")
        
        generate_and_save_geometry(h11, cy, points, simplices, filepath)
        return f"Saved {filepath}"
    except Exception as e:
        return f"Error on np_{tri_idx+1:07d}: {e}"
    
def run_batch(h11, n_polytopes, base_dir, n_cores=None):
    poly_list = fetch_polytopes(h11, limit=n_polytopes * 8, lattice="N", favorable=True)
    if not poly_list:
        print(f"No favorable polytopes found for h11={h11}")
        return
        
    # 1. Package the tasks (extracting raw points)
    tasks = []
    for tri_idx, poly in enumerate(poly_list[:n_polytopes]):
        points = np.array(poly.points(), dtype=int)
        tasks.append((tri_idx, points, h11, base_dir))
        
    # 2. Map the tasks across multiple cores
    print(f"Launching {len(tasks)} tasks across {n_cores if n_cores else 'all available'} cores...")
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # executor.map handles the queueing and returns results as they finish
        for result in executor.map(process_single_polytope, tasks):
            print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CYTools geometry data for Julia.")
    parser.add_argument("--h11_min", type=int, default=4, help="Starting h11 value.")
    parser.add_argument("--h11_max", type=int, default=4, help="Ending h11 value (inclusive).")
    parser.add_argument("--n", type=int, default=1, help="Number of polytopes to process per h11.")
    parser.add_argument("--outdir", type=str, default=".", help="Base directory to save the data.")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use (default: all).")
    args = parser.parse_args()
    
    # Sanity check: if a user passes only h11_min, make it a single-value run
    if args.h11_max < args.h11_min:
        args.h11_max = args.h11_min
        
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Starting generation for h11 in range [{args.h11_min}, {args.h11_max}], n={args.n} each. Saving to: {args.outdir}")
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Loop over the range of h11 values
    for h in range(args.h11_min, args.h11_max + 1):
        print(f"\n>>> Processing h11 = {h} <<<")
        run_batch(h11=h, n_polytopes=args.n, base_dir=args.outdir, n_cores=args.cores)
        
    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    total_cys = (args.h11_max - args.h11_min + 1) * args.n
    
    print("\n" + "="*30)
    print("      BENCHMARK RESULTS      ")
    print("="*30)
    print(f"Total CYs:   {total_cys}")
    print(f"Total Time:  {end_time - start_time:.3f} seconds")
    print(f"Peak Memory: {peak_mem / (1024 * 1024):.2f} MB")
    if total_cys > 0:
        print(f"Time/CY:     {(end_time - start_time) / total_cys:.3f} seconds")
    print("="*30)

# python generate_geometric_data.py --h11_min 4 --h11_max 20 --n 5 --outdir test_data --cores 8
