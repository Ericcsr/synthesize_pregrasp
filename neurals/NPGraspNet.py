import copy
from unittest.mock import NonCallableMagicMock
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from neurals.dex_grasp_net import DexGraspNetModel
from neurals.network import ScoreFunction, LargeScoreFunction

class NPGraspNet(nn.Module):
    def __init__(self, opt=None, pred_fingers=[2], extra_cond_fingers=[0,1], mode="only_score", has_distance_field=False, device="cuda:1"):
        """
        mode: 
        "only_score": only use score function, no encoder or decoder
        "decoder": decoder + score function
        "full": decoder + encoder + small score function
        """
        super(NPGraspNet, self).__init__()
        self.pred_fingers = pred_fingers
        self.extra_cond_fingers = extra_cond_fingers
        self.num_pred_fingers = len(pred_fingers)
        self.num_extra_fingers = len(extra_cond_fingers)
        self.device = torch.device(device)
        self.gpu_id = int(device[-1]) if device != "cpu" else -1
        self.mode = mode
        assert(self.mode in ["only_score", "decoder", "full"])
        
        if not self.mode == "only_score":
            self.dex_grasp_net = DexGraspNetModel(opt, pred_base=False,
                                                  pred_fingers=self.pred_fingers,
                                                  extra_cond_fingers=extra_cond_fingers,
                                                  gpu_id = self.gpu_id)
            self.dex_grasp_net.eval()
        
        if self.mode == "full":
            self.score_function = ScoreFunction(self.dex_grasp_net.get_latent_size()).cuda(device=self.gpu_id)
        elif self.mode == "decoder":
            self.score_function = LargeScoreFunction(self.num_extra_fingers + self.num_pred_fingers, has_distance_field=has_distance_field).cuda(device=self.gpu_id)
        else:
            if self.gpu_id == -1:
                self.score_function = LargeScoreFunction(self.num_extra_fingers, latent_dim=20, has_distance_field=has_distance_field).cpu()
            else:
                self.score_function = LargeScoreFunction(self.num_extra_fingers, latent_dim=20, has_distance_field=has_distance_field).cuda(device=self.gpu_id)
        self.score_function.eval()

    def pred_score(self,pcd, extra_cond, latent=None, distance_field=None):
        """
        pcd: [1, N_points, 3]
        extra_cond: Assume flatten vector
        """
        df_th = None
        if self.gpu_id != -1:
            pcd_np = torch.from_numpy(np.asarray(pcd.points)).view(1, -1, 3).cuda(device=self.gpu_id).float()
            if distance_field is not None:
                df_th = distance_field.view(1,-1,1).cuda(device = self.gpu_id).float()
        else:
            pcd_np = torch.from_numpy(np.asarray(pcd.points)).view(1, -1, 3).float()
            if distance_field is not None:
                df_th = distance_field.view(1, -1, 1).view(1,-1,1).float()
        if self.mode != "only_score":
            latent = latent.cuda(device=self.gpu_id).view(1,-1).float()
        if self.gpu_id != -1:
            extra_cond_th = torch.from_numpy(extra_cond).cuda(device=self.gpu_id).float()
        else:
            extra_cond_th = torch.from_numpy(extra_cond).float()
        if self.mode == "only_score":
            score = self.score_function.pred_score(pcd_np, extra_cond_th, dist_field=df_th)[0].cpu()
            return score, None
        else:
            ans, _, _ = self.dex_grasp_net.generate_grasps(pcd_np, z=latent, extra_cond=extra_cond_th) # Should be [1,9]
                # Need to do projection based on current pointcloud input
            full_grasp = torch.vstack([extra_cond_th.view(self.num_extra_fingers,3), ans.view(self.num_pred_fingers, 3)])
            full_grasp = self.project_grasp(pcd_np, full_grasp).view(1, -1)
            if self.mode == "full":
                with torch.no_grad():
                    # Need to encode again， note that here the latent state only have 3 dimension which is really bad
                    latent2 = self.dex_grasp_net.encode(pcd_np, full_grasp)
                    score = self.score_function.pred_score(latent2)[0].cpu()
            else:
                with torch.no_grad():
                    score = self.score_function.pred_score(pcd_np, full_grasp, distance_field=df_th)[0].cpu()
            return score, self._parse_ans(ans, extra_cond)

    def pred_grasp(self, pcd_points, extra_cond, num_grasps=10):
        """
        pcd: Assume pcd is open3d pcd file
        extra_cond: assume it is a [2,3] np.array contains position of thumb and index finger
        num_grasps: number of samples that we use
        """
        # Down sample pointcloud if needed
        assert(len(pcd_points) > 1024)
        downsample_idx = np.random.choice(len(pcd_points), 1024, replace=False)
        pcd_points = pcd_points[downsample_idx]
        pcd_th = torch.from_numpy(pcd_points).view(1, -1, 3).cuda(device=self.gpu_id).float()
        extra_cond_th = torch.from_numpy(extra_cond).view(1,-1).cuda(device=self.gpu_id).float()
        grasps = []
        for _ in range(num_grasps):
            latent = self.sample_latent(1)
            ans, _, _ = self.dex_grasp_net.generate_grasps(pcd_th, z=latent, extra_cond=extra_cond_th)
            full_grasp = torch.vstack([extra_cond_th.view(self.num_extra_fingers,3), ans.view(self.num_pred_fingers,3)])
            grasps.append(full_grasp.cpu().numpy())
        return grasps

    def _parse_ans(self, ans, extra_cond=None):
        """
        Here assume ans is 
        """
        ans = ans.squeeze().cpu().numpy()
        finger_pose = np.zeros((self.num_pred_fingers+self.num_extra_fingers, 3))
        for i in range(self.num_pred_fingers):
            finger_pose[self.pred_fingers[i]] = ans[i*3:(i+1)*3]
        if not (extra_cond is None):
            extra_cond = extra_cond.squeeze()
            for j in range(self.num_extra_fingers):
                finger_pose[self.extra_cond_fingers[j]] = extra_cond[j*3:(j+1)*3]
        return finger_pose

    def load_dex_grasp_net(self, model_path):
        state_dict = torch.load(model_path, map_location=torch.device(f"cuda:{self.gpu_id}"))
        self.dex_grasp_net.net.load_state_dict(state_dict["model_state_dict"])

    def load_score_function(self, model_path):
        if self.gpu_id != -1:
            self.score_function.load_state_dict(torch.load(model_path, map_location=torch.device(f"cuda:{self.gpu_id}")))
        else:
            self.score_function.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    def sample_latent(self, batch_size):
        return self.dex_grasp_net.sample_latent(batch_size)

    def create_kd_tree(self,pcd):
        self.kd_tree = o3d.geometry.KDTreeFlann(pcd)
    # Currently just find the closest point on the pointcloud

    def project_grasp(self, pcd_tensor, grasps):
        grasps = copy.deepcopy(grasps.view(-1,3)).cpu().numpy()
        pcd_tensor = pcd_tensor[0].cpu().numpy() #[N_points, 3]
        for i, grasp in enumerate(grasps):
            index = self.kd_tree.search_knn_vector_3d(grasp, 1)[1]
            grasps[i] = pcd_tensor[index]
        return torch.from_numpy(grasps).cuda(device=self.gpu_id)

    def get_latent_size(self):
        return self.dex_grasp_net.get_latent_size()
        
            


