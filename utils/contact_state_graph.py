import numpy as np
import copy

# Currently assume a feasible contact state must have at least a region for each finger
# Currently region can be defined by number
# Finger order: 0:thumb 1:index 2:middle 3:ring hence region [2,6,3,7]
# Need a mechanism that parse the region id to region a function input is action + region id output is finger tip pose. each object should have a unique action parser
class ContactStateGraph:
    def __init__(self, states):
        """
        states: np.array of region id
        """
        self.states = states
        self.state_ids = list(range(len(self.states)))
        self.graph_adj_list = self.build_graph(self.states, self.state_ids)

    # Represent the graph as adjacency list, each entry should have number indicating edge type
    def build_graph(self, states, state_ids):
        # initialize adjacency list
        graph_adj_list = {}
            
        # Build the list
        for i in range(len(self.state_ids)):
            i_id = state_ids[i]
            graph_adj_list[i_id] = []
            for j in range(len(state_ids)):
                j_id = state_ids[j]
                common_regions = (states[i_id] == states[j_id])
                num_common_regions = common_regions.sum()
                if num_common_regions != 0:
                    graph_adj_list[i_id].append((j_id, common_regions, num_common_regions))
        return graph_adj_list

    def getNeighbors(self,state_id):
        return self.graph_adj_list[state_id]

    def getState(self, state_id):
        return self.states[state_id]

    # return all paths as well as their weights (absolute) maybe recursive?
    def _getPathFromState(self, state_id, steps, total_weight=1, current_path=[]):
        neighbors = self.getNeighbors(state_id)
        weights = np.array([n[2] for n in neighbors])
        weights = weights/weights.sum()
        current_path.append(state_id)
        for i,neighbor in enumerate(neighbors):
            if steps != 1:
                self._getPathFromState(neighbor[0],steps-1, total_weight*weights[i], current_path=copy.deepcopy(current_path))
            else:
                self.paths.append(current_path+[neighbor[0]])
                self.weights.append(total_weight*weights[i]) # Final weights

    def getPathFromState(self,state_id, steps):
        self.paths = []
        self.weights = []
        self._getPathFromState(state_id, steps,current_path=[])
        return self.paths, self.weights

    def samplePathFromState(self, state_id, steps, n_paths):
        self.paths = []
        self.weights = []
        for i in range(n_paths):
            path = [state_id]
            cursor = state_id
            total_weight = 1
            for i in range(steps):
                neighbors = self.getNeighbors(cursor)
                neighbors_id = [n[0] for n in neighbors]
                weights = np.array([n[2] for n in neighbors], dtype=np.float32)
                weights /= weights.sum()
                idx = np.random.choice(np.arange(len(weights)), p=weights)
                cursor = neighbors_id[idx]
                path.append(cursor)
                total_weight *= weights[idx]
            self.paths.append(path)
            self.weights.append(total_weight)
        return self.paths, self.weights
            
    
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

    csg = ContactStateGraph(states)
    paths, weights = csg.getPathFromState(0,4)
    print(paths)
    print(sum(weights))