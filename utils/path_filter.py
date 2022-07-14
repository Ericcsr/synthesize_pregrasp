import numpy as np
from compute_dyn_feasible_contacts import check_dyn_feasible

def filter_paths(paths, csg, region, n_paths=20):
    filtered_path = []
    for i in range(len(paths)):
        path = paths[i] # Here path is already array of state arrays
        last_regions_id = csg.getState(path[-1]) - 1
        last_contact_points = region.centers[last_regions_id]
        last_contact_normals = region.surface_norm[last_regions_id]
        if not (check_dyn_feasible(last_contact_points, last_contact_normals) is None):
            filtered_path.append(path)
        if len(filtered_path) == n_paths:
            break
    return np.asarray(filtered_path)
    
