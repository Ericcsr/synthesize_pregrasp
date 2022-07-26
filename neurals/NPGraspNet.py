import copy
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from neurals.dex_grasp_net import DexGraspNetModel
from neurals.network import ScoreFunction, LargeScoreFunction

class NPGraspNet(nn.Module):
    def __init__(self, opt=None, pred_fingers=[2], extra_cond_fingers=[0,1], use_large_model=False, device="cuda:0"):
        super(NPGraspNet, self).__init__()
        self.pred_fingers = pred_fingers
        self.extra_cond_fingers = extra_cond_fingers
        self.num_pred_fingers = len(pred_fingers)
        self.num_extra_fingers = len(extra_cond_fingers)
        self.device = torch.device(device)
        self.use_large_model = use_large_model
        
        self.dex_grasp_net = DexGraspNetModel(opt, pred_base=False,
                                              pred_fingers=self.pred_fingers,
                                              extra_cond_fingers=extra_cond_fingers)
        if not self.use_large_model:
            self.score_function = ScoreFunction(self.dex_grasp_net.get_latent_size()).to(device)
        else:
            self.score_function = LargeScoreFunction(self.num_extra_fingers + self.num_pred_fingers)

    def pred_score(self,pcd, latent, extra_cond=None):
        """
        pcd: [1, N_points, 3]
        extra_cond: Assume flatten vector
        """
        pcd_np = torch.from_numpy(np.asarray(pcd.points)).view(1, -1, 3)
        pcd_np.to(self.device)
        latent.to(self.device)
        if not (extra_cond is None):
            extra_cond.to(self.device)
            pcd_np = torch.cat(
            (pcd_np, extra_cond.unsqueeze(1).expand(-1,pcd_np.shape[1], -1)),
            -1).transpose(-1, 1).contiguous() # Extend the point cloud to include extra features
        ans, _, _ = self.dex_grasp_net.generate_grasps(pcd_np, z=latent, extra_cond=extra_cond) # Should be [1,9]
            # Need to do projection based on current pointcloud input
        ans = self.project_grasp(ans)
        if self.use_large_model:
            with torch.no_grad():
                # Need to encode again， note that here the latent state only have 3 dimension which is really bad
                latent2 = self.dex_grasp_net.encode(pcd, ans)
                score = self.score_function.pred_score(latent2)[0].cpu()
        else:
            with torch.no_grad():
                score = self.score_function.pred_score(pcd, ans)
        return score, self._parse_ans(ans, extra_cond)

    def _parse_ans(self, ans, extra_cond=None):
        """
        Here assume ans is 
        """
        ans = ans.squeeze().cpu().numpy()
        finger_pose = np.zeros((self.num_fingers, 3))
        for i in range(self.num_pred_fingers):
            finger_pose[self.pred_fingers[i]] = ans[i*3:(i+1)*3]
        if not (extra_cond is None):
            extra_cond = extra_cond.squeeze().cpu()
            for j in range(self.num_extra_fingers):
                finger_pose[self.extra_cond_fingers[j]] = extra_cond[j*3:(j+1)*3]
        return finger_pose

    def load_dex_grasp_net(self, model_path):
        state_dict = torch.load(model_path)
        self.dex_grasp_net.net.load_state_dict(state_dict["model_state_dict"])

    def load_score_function(self, model_path):
        self.score_function.load_state_dict(torch.load(model_path))

    def sample_latent(self, batch_size):
        return self.dex_grasp_net.sample_latent(batch_size)

    def create_kd_tree(self,pcd):
        self.kd_tree = o3d.geometry.KDTreeFlann(pcd)
    # Currently just find the closest point on the pointcloud

    def project_grasp(self, pcd_tensor, grasps):
        grasps = copy.deepcopy(grasps.view(-1,3))
        pcd_tensor = pcd_tensor[0] #[N_points, 3]
        for i, grasp in enumerate(grasps):
            index = int(self.kd_tree.search_knn_vector_3d(grasp, 1)[1])
            grasps[i] = pcd_tensor[index]
        return grasps
        
            


