import numpy as np
import copy
import itertools
# Currently assume a feasible contact state must have at least a region for each finger
# Currently region can be defined by number
# Finger order: 0:thumb 1:index 2:middle 3:ring hence region [2,6,3,7]
# Need a mechanism that parse the region id to region a function input is action + region id output is finger tip pose. each object should have a unique action parser
class ContactStateEmbedding:
    def __init__(self, states, distance_table, beta):
        """
        states: np.array of region id
        distance_table: should be something similar to adj matrix
        """
        self.states = states
        self.beta = beta
        self.distance_table = distance_table # Store geodesic distance of high res mesh.
        self.state_ids = list(range(len(self.states)))
    
    # The distance table can be hand calculated or calculated by other package
    def distance(self,p_id, q_id):
        p_state = self.states[p_id]
        q_state = self.states[q_id]
        uncommon_regions = (p_state != q_state)
        num_uncommon_regions = uncommon_regions.sum()
        p_unique_regions = p_state[uncommon_regions]
        q_unique_regions = q_state[uncommon_regions]
        total_surface_distance = 0
        for p_unique, q_unique in zip(p_unique_regions, q_unique_regions):
            total_surface_distance += self.distance_table[p_unique, q_unique]
        return self.beta * (num_uncommon_regions) + (1-self.beta) * total_surface_distance

    def getState(self, state_id):
        return self.states[state_id]

    def getPathFromState(self,start_state_id, steps):
        paths = list(itertools.product(self.state_ids,steps-1))
        distance = []
        # Sort path by path distance, shortest first
        for path in paths:
            path.insert(0, start_state_id)
            cum_dist = 0
            for i in range(steps-1):
                cum_dist += self.distance(path[i],path[i+1])
            distance.append(cum_dist)
        sorted_idx = np.argsort(distance)
        return np.asarray(paths)[sorted_idx], np.asarray(distance)[sorted_idx]
            
    
if __name__ == "__main__":
    states = []
    states.append(np.array([5, 27, 26, 25]))
    states.append(np.array([5, 23, 26, 29]))
    states.append(np.array([5, 25, 26, 27]))
    states.append(np.array([5, 29, 26, 23]))
    states.append(np.array([26, 4, 5, 6]))
    states.append(np.array([26, 8, 5, 2]))
    states.append(np.array([26, 6, 5, 4]))
    states.append(np.array([26, 2, 5, 8]))

    csg = ContactStateEmbedding(states)
    paths, weights = csg.getPathFromState(0,4)
    print(paths)
    print(sum(weights))