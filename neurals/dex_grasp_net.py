import torch
import neurals.network as network
import numpy as np
import os
from os.path import join
import time

class DexGraspNetModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    Assume opt are dict
    e.g.,
    --dataset_mode -> sampling / evaluation)
    """
    def __init__(self, opt, pred_base=True, pred_fingers=[0, 1, 2], extra_cond_fingers=[]):
        if isinstance(opt, dict):
            self.opt = opt
        else:
            opt = opt.__dict__
            self.opt = opt
        self.gpu_ids = opt['gpu_ids'] #opt.gpu_ids
        self.is_train = opt['is_train'] #opt.is_train
        self.pred_base = pred_base
        self.num_in_feats = 3 * len(pred_fingers) + 3 # feature numner + 3
        if pred_base:
            self.num_in_feats += 12
        self.extra_cond_dim = len(extra_cond_fingers) * 3
        self.extra_cond_fingers = extra_cond_fingers
        self.pred_fingers = pred_fingers
        self.num_pred_fingers = len(pred_fingers)
        assert torch.cuda.device_count()>=1
        if self.gpu_ids and self.gpu_ids[0] >= torch.cuda.device_count():
            self.gpu_ids[0] = torch.cuda.device_count() - 1
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        if self.opt['is_train']: # self.opt.is_train:
            if self.opt["override_saved"]:
                #self.save_dir = join(opt.checkpoints_dir, opt.name)
                self.save_dir = join(opt["checkpoints_dir"], opt["name"])
            else:
                #self.save_dir = join(opt.checkpoints_dir, opt.name, str(time.time()))
                self.save_dir = join(opt["checkpoints_dir"], opt["name"], str(time.time()))
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                #assert self.opt.override_saved
                assert self.opt["override_saved"]
        # else:
        #     if opt.store_timestamp is None:
        #         self.save_dir = join(opt.checkpoints_dir, opt.name)
        #     else:
        #         self.save_dir = join(opt.checkpoints_dir, opt.name, opt.store_timestamp)
        # self.save_dir_no_time = join(opt.checkpoints_dir, opt.name)
        else:
            if not(opt["store_timestamp"] is None):
                self.save_dir = join(opt["checkpoints_dir"], opt["name"], opt["store_timestamp"])
            else:
                self.save_dir = join(opt["checkpoints_dir"], opt["name"])
        self.optimizer = None
        self.loss_train = None
        self.pcs = None
        # load/define networks
        # TODO: Need to check lower-level dependency on opt
        # self.net = network.define_graspgen(opt, self.gpu_ids, opt.arch,
        #                                       opt.init_type, opt.init_gain,
        #                                       self.pred_base, 
        #                                       num_in_feats = self.num_in_feats,
        #                                       extra_cond_dim = self.extra_cond_dim, 
        #                                       num_pred_fingers=self.num_pred_fingers,
        #                                       device=self.device)
        self.net = network.define_graspgen(opt, self.gpu_ids, opt["arch"],
                                            opt["init_type"], opt["init_gain"],
                                            self.pred_base,
                                            num_in_feats=self.num_in_feats,
                                            extra_cond_dim=self.extra_cond_dim,
                                            num_pred_fingers=self.num_pred_fingers,
                                            device=self.device)

        self.criterion = network.define_loss(opt, pred_base)

        self.confidence_loss_train = None
        #if self.opt.arch == "vae":
        if self.opt["arch"] == "vae":
            self.kl_loss_train = None
            self.fingertip_pos_loss_train = None
            if self.pred_base:
                self.base_pos_loss_train = None
                self.base_orn_loss_train = None
            
        # elif self.opt.arch == "gan":
        #     self.reconstruction_loss = None
        # else:
        #     self.classification_loss = None

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr= opt["lr"], #opt.lr,
                                              #betas=(opt.beta1, 0.999))
                                              betas=(opt["beta1"], 0.999))
            self.scheduler = network.get_scheduler(self.optimizer, opt)
        # if not self.is_train or opt.continue_train:
        #     self.load_network(opt.which_epoch, self.is_train, opt.which_timestamp)
        if (not self.is_train or opt["continue_train"]) and not opt["force_skip_load"]:
            self.load_network(opt["which_epoch"], self.is_train, opt["which_timestamp"])
        # Weight on distance between grasp points
        '''
        3D [base_position_weight, orientation_weight, fingertip_position_weight]
        '''
        if self.pred_base:
            self.distance_weight = torch.ones(3, device=self.device, dtype=torch.float32)
            # position weights scaled by 100
            self.distance_weight[0] *= 2.
            self.distance_weight[1] *= 0.2
            self.distance_weight[2] *= 10.
        else:
            self.distance_weight = torch.ones(1,device=self.device, dtype=torch.float32)
            self.distance_weight[0] *= 10

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def set_input(self, data):
        # The default collate_fn already converts np arrays to tensors
        input_pcs = data['point_cloud'].contiguous().float()
        if self.pred_base:
            input_grasps = data['grasp'].float() # Recall this is hstack([self.base_position, self.base_rotation_matrix_flattened])
            input_fingertip_positions = data['fingertip_normals'].float()[:,:,:3].reshape(-1,9) # ignore the normal vectors
            # concatenate the tensors on the second axis to get nx32 tensor
            targets = torch.cat([input_grasps, input_fingertip_positions], axis=1) # * x 21
        else:
            targets = data['fingertip_pos'][:,self.finger_to_idx(self.pred_fingers)].float()
            extra_cond = data['fingertip_pos'][:,self.finger_to_idx(self.extra_cond_fingers)].float()
            self.extra_cond = extra_cond.to(self.device).requires_grad_(self.is_train)
        self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)
        self.targets = targets.to(self.device).requires_grad_(self.is_train)

    def set_input_test(self, data):
        # The default collate_fn already converts np arrays to tensors
        input_pcs = data['point_cloud'].contiguous().float()
        if self.pred_base:
            input_grasps = data['grasp'].float() # Recall this is hstack([self.base_position, self.base_rotation_matrix_flattened])
            input_fingertip_positions = data['fingertip_normals'].float()[:,:,:3].reshape(-1,9) # ignore the normal vectors
            # concatenate the tensors on the second axis to get nx32 tensor
            targets = torch.cat([input_grasps, input_fingertip_positions], axis=1)
        else:
            targets = data['fingertip_pos'][:,self.finger_to_idx(self.pred_fingers)].float()
            extra_cond = data['fingertip_pos'][:,self.finger_to_idx(self.extra_cond_fingers)].float()
            self.extra_cond_test = extra_cond.to(self.device).requires_grad_(False)
        self.pcs_test = input_pcs.to(self.device).requires_grad_(False)
        self.targets_test = targets.to(self.device).requires_grad_(False)

    def generate_grasps(self, pcs, z=None, extra_cond=None, return_raw_q = True):
        with torch.no_grad():
            ans = self.net.generate_grasps(pcs, z=z, extra_cond=extra_cond)
            if return_raw_q:
                return ans
            else:
                raise NotImplementedError
                # return (*convert_network_output_to_hand_conf(ans[0]), *ans[1:])

    def forward(self):
        return self.net(self.pcs, self.targets, self.extra_cond if self.extra_cond_dim!= 0 else None)

    def encode(self, pcs, grasp_pts, extra_cond=None):
        # Append the feature at the end of each points
        if extra_cond is None:
            input_features = torch.cat(
                (pcs, grasp_pts.unsqueeze(1).expand(-1,pcs.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
        else:
            input_features = torch.cat(
                (pcs, grasp_pts.unsqueeze(1).expand(-1, pcs.shape[1], -1), 
                 extra_cond.unsqueeze(1).expand(-1, pcs.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
        latent_input = self.net.encode(pcs.float(), input_features.float())
        return self.net.latent_space[0](latent_input)

    def _compute_loss(self, out, targets):
        predicted_grasp, confidence, mu, logvar = out
        #targets = targets[:,self.finger_to_idx(self.pred_fingers)]
        # Reconstruction loss is control_point_l1_loss
        if self.pred_base:
            base_pos_loss_train, base_orn_loss_train, fingertip_pos_loss_train = self.criterion[1](
                predicted_grasp,
                targets,
                confidence=confidence,
                confidence_weight=self.opt["confidence_weight"],
                device=self.device,
                distance_weight=self.distance_weight)
        else:
            fingertip_pos_loss_train = self.criterion[1](
                predicted_grasp,
                targets,
                distance_weight=self.distance_weight)
        kl_loss = self.opt["kl_loss_weight"] * self.criterion[0](
                    mu, logvar, device=self.device)
        if self.pred_base:
            total_loss = kl_loss + base_pos_loss_train + base_orn_loss_train + fingertip_pos_loss_train
            return total_loss, kl_loss, base_pos_loss_train, base_orn_loss_train, fingertip_pos_loss_train
        else:
            total_loss = kl_loss + fingertip_pos_loss_train
            return total_loss, kl_loss, fingertip_pos_loss_train
        
    def compute_loss_on_test_data(self):
        with torch.no_grad():
            # Do a forward pass
            targets_test = self.targets_test
            out = self.net(self.pcs_test, targets_test, self.extra_cond_test if self.extra_cond_dim!=0 else None)
            if self.opt["arch"] == 'vae' and self.pred_base:
                self.loss_test, self.kl_loss_test, self.base_pos_loss_test, self.base_orn_loss_test, self.fingertip_pos_loss_test = \
                    self._compute_loss(out, targets_test)
            elif self.opt["arch"] == "vae" and not self.pred_base:
                self.loss_test, self.kl_loss_test, self.fingertip_pos_loss_test = \
                    self._compute_loss(out, targets_test)
            else:
                raise NotImplementedError

    def backward(self, out):
        if self.opt["arch"] == 'vae' and self.pred_base:
            self.loss_train, self.kl_loss_train, self.base_pos_loss_train, self.base_orn_loss_train, self.fingertip_pos_loss_train = \
                self._compute_loss(out, self.targets)
        elif self.opt["arch"] == 'vae' and not self.pred_base:
            self.loss_train, self.kl_loss_train, self.fingertip_pos_loss_train = \
                self._compute_loss(out, self.targets)
        self.loss_train.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    def finger_to_idx(self,fingers):
        idxs = []
        for finger in fingers:
            idxs += [finger,finger+1, finger+2]
        return idxs

    def sample_latent(self, batch_size):
        return self.net.sample_latent(batch_size)


##################

    def load_network(self, which_epoch, train=True, which_timestamp=''):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        if which_timestamp == '':
            load_path = join(self.save_dir_no_time, save_filename)
        else:
            load_path = join(self.save_dir_no_time, which_timestamp, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        checkpoint = torch.load(load_path, map_location=self.device)
        if hasattr(checkpoint['model_state_dict'], '_metadata'):
            del checkpoint['model_state_dict']._metadata
        net.load_state_dict(checkpoint['model_state_dict'])
        if train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.opt["epoch_count"] = checkpoint["epoch"]
        else:
            net.eval()

    def load_network_test(self, network_path):
        if isinstance(self.net, torch.nn.DataParallel):
            self.net.module.load_state_dict(torch.load(network_path))
        else:
            self.net.load_state_dict(torch.load(network_path))

    def save_network(self, net_name, epoch_num):
        """save model to disk"""
        save_filename = '%s_net.pth' % (net_name)
        save_path = join(self.save_dir, save_filename)
        torch.save(
            {
                'epoch': epoch_num + 1,
                'model_state_dict': self.net.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.net.cuda(self.gpu_ids[0])

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_latent_size(self):
        return self.net.get_latent_size()
    