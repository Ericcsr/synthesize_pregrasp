# Adapted from pytorch_6dof-graspnet (https://github.com/jsll/pytorch_6dof-graspnet)
from neurals import losses
import utils.math_utils as math_utils

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import pointnet2_ops.pointnet2_modules as pointnet2


def get_scheduler(optimizer, opt):
    #if opt.lr_policy == 'lambda':
    if opt["lr_policy"] == "lambda":
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(
            #     0, epoch + 1 + 1 - opt.niter) / float(opt.niter_decay + 1)
            lr_l = 1.0 - max(0, epoch+2-opt["niter"])/float(opt["niter_decay"]+1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    #elif opt.lr_policy == 'step':
    elif opt["lr_policy"] == "step":
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size= opt["lr_decay_iters"], # opt.lr_decay_iters,
                                        gamma=0.1)
    #elif opt.lr_policy == 'plateau':
    elif opt["lr_policy"] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    else:
        # return NotImplementedError(
        #     'learning rate policy [%s] is not implemented', opt.lr_policy)
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", opt["lr_policy"])
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type, init_gain=0.02, gpu_ids=None):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        #net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_graspgen(opt, gpu_ids, arch, init_type, init_gain, pred_base, num_in_feats, extra_cond_dim, num_pred_fingers, device):
    net = None
    if arch == 'vae':
        # net = GraspSamplerVAE(opt.model_scale, opt.pointnet_radius,
        #                       opt.pointnet_nclusters, opt.latent_size, 
        #                       device, num_in_feats=num_in_feats,pred_base=pred_base, 
        #                       extra_cond_dim=extra_cond_dim, num_pred_fingers=num_pred_fingers)
        net = GraspSamplerVAE(opt["model_scale"], opt["pointnet_radius"],
                              opt["pointnet_nclusters"], opt["latent_size"],
                              device, num_in_feats=num_in_feats,pred_base=pred_base,
                              extra_cond_dim=extra_cond_dim, num_pred_fingers=num_pred_fingers)
    else:
        raise NotImplementedError('model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_loss(opt,pred_base=True):
    #if opt.arch == 'vae' and pred_base:
    if opt["arch"] == "vae" and pred_base:
        kl_loss = losses.kl_divergence
        reconstruction_loss = losses.base_pose_and_fingertip_positions_l1_loss
        return kl_loss, reconstruction_loss
    #elif opt.arch == 'vae' and not pred_base:
    elif opt["arch"] == "vae" and not pred_base:
        kl_loss = losses.kl_divergence
        reconstruction_loss = losses.fingertip_positions_l1_loss
        return kl_loss, reconstruction_loss

    # elif opt.arch == 'gan':
    #     reconstruction_loss = losses.min_distance_loss
    #     return reconstruction_loss
    # elif opt.arch == 'evaluator':
    #     loss = losses.classification_with_confidence_loss
    #     return loss
    else:
        raise NotImplementedError("Loss not found")

class GraspSampler(nn.Module):
    def __init__(self, latent_size, device):
        super(GraspSampler, self).__init__()
        self.latent_size = latent_size
        self.device = device

    def create_decoder(self, model_scale, pointnet_radius, pointnet_nclusters,
                       num_input_features, pred_base=True, num_pred_fingers=3):
        """
        Creates the CVAE decoder. The number of input features for the decoder
        is 3+latent space where 3 represents the x, y, z position of the point-cloud
        The output is 18 dof: 3D base pose + 6D base rotation + 3x3D fingertip positions
        :param model_scale:
        :param pointnet_radius:
        :param pointnet_nclusters:
        :param num_input_features:
        """
        # The number of input features for the decoder is 3+latent space where 3
        # represents the x, y, z position of the point-cloud

        self.decoder = base_network(pointnet_radius, pointnet_nclusters,
                                    model_scale, num_input_features)
        # Features to predict
        if pred_base:
            self.base_pos_layer = nn.Linear(model_scale * 512, 3)
            self.base_orn_layer = nn.Linear(model_scale * 512, 6)
            self.pred_base = True
        else:
            self.pred_base = False
        self.fingertip_locations_layer = nn.Linear(model_scale * 512, 3 * num_pred_fingers) # 3x3D pose #
        # self.confidence = nn.Linear(model_scale * 512, 1)

    def decode(self, xyz, z, extra_cond=None):
        """
        Decoder layer.
        :param xyz:
        :param z:
        :return: (predicted_grasp, confidence)
        predicted_grasp has shape (batch_size, 18): 3D base pose + 6D base rotation + 3x3D fingertip positions
        confidence has shape batch_size
        """
        if extra_cond is None:
            xyz_features = self.concatenate_z_with_pc(xyz,
                                                  z).transpose(-1,
                                                               1).contiguous()
        else:
            z = torch.hstack([z, extra_cond])
            xyz_features = self.concatenate_z_with_pc(xyz,
                                                  z).transpose(-1,
                                                               1).contiguous()
        for module in self.decoder[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        x = self.decoder[1](xyz_features.squeeze(-1))
        # Process the orientation prediction
        if self.pred_base:    
            base_rotation_matrix = math_utils.convert_6d_rotation_to_flattened_matrix_torch(self.base_orn_layer(x))
            predicted_grasp = torch.cat(
                (self.base_pos_layer(x),
                base_rotation_matrix,
                self.fingertip_locations_layer(x)), -1)
        else:
            predicted_grasp = self.fingertip_locations_layer(x)
        return predicted_grasp, None #torch.sigmoid(self.confidence(x)).squeeze()

    def concatenate_z_with_pc(self, pc, z):
        z.unsqueeze_(1)
        z = z.expand(-1, pc.shape[1], -1)
        return torch.cat((pc, z), -1)

    def get_latent_size(self):
        return self.latent_size


class GraspSamplerVAE(GraspSampler):
    """Network for learning a generative VAE grasp-sampler
    """
    def __init__(self,
                 model_scale,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 latent_size=3,
                 device="cuda",
                 num_in_feats=24,
                 pred_base=True,
                 extra_cond_dim=0,
                 num_pred_fingers=3):
        self.device = device
        self.pred_base = pred_base
        super(GraspSamplerVAE, self).__init__(latent_size, device)
        self.create_encoder(model_scale, pointnet_radius, pointnet_nclusters, n_input_feats=num_in_feats+extra_cond_dim)

        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                            latent_size + 3 + extra_cond_dim, pred_base=pred_base, num_pred_fingers=num_pred_fingers)
        self.create_bottleneck(model_scale * 512, latent_size)

    def create_encoder(
            self,
            model_scale,
            pointnet_radius,
            pointnet_nclusters,
            n_input_feats
    ):
        # The number of input features for the encoder is 24:
        # (x,y,z) of points in point cloud +
        # 3D base position + 9D base rotation matrix + 3x3D fingertip positions
        # Recall the signature is
        # base_network(pointnet_radius, pointnet_nclusters, scale, in_features)
        self.encoder = base_network(pointnet_radius, pointnet_nclusters,
                                    model_scale, n_input_feats)

    # Map real latent state to mean and variance
    def create_bottleneck(self, input_size, latent_size):
        mu = nn.Linear(input_size, latent_size)
        logvar = nn.Linear(input_size, latent_size)
        self.latent_space = nn.ModuleList([mu, logvar])

    # Here XYZ means pointclouds, which is also the condition, should be more than pointcloud
    # Output dimension can be accessed via get_latent_size()
    def encode(self, xyz, xyz_features):
        for module in self.encoder[0]: # Encoder 0 are PointNet layers
            xyz, xyz_features = module(xyz, xyz_features)
        return self.encoder[1](xyz_features.squeeze(-1))

    def bottleneck(self, z):
        return self.latent_space[0](z), self.latent_space[1](z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pcs, grasps, extra_cond=None):# train=True):
        return self.forward_train(pcs, grasps, extra_cond)

    def forward_train(self, pcs, grasps, extra_cond=None):
        """
        Function for training the network
        :param input:
        :return:
        """
        if extra_cond is None:
            input_features = torch.cat(
                (pcs, grasps.unsqueeze(1).expand(-1, pcs.shape[1], -1)),
                -1).transpose(-1, 1).contiguous() # (batch_size*(3+in_feat_dim)*256)
        else:
            #print(grasps.shape, extra_cond.shape, pcs.shape)
            input_features = torch.cat(
                (pcs, grasps.unsqueeze(1).expand(-1, pcs.shape[1], -1), 
                 extra_cond.unsqueeze(1).expand(-1, pcs.shape[1], -1)), -1).transpose(-1,1).contiguous() # # (batch_size*(3+extra_dim+in_feat_dim)*256)
        z = self.encode(pcs, input_features) # Latent variable
        mu, logvar = self.bottleneck(z)
        z = self.reparameterize(mu, logvar)
        predicted_grasp, confidence = self.decode(pcs, z, extra_cond)
        return predicted_grasp, confidence, mu, logvar

    def sample_latent(self, batch_size):
        # torch.randn samples from N(0, I)
        return torch.randn(batch_size, self.latent_size)

    def generate_grasps(self, pc, z=None, extra_cond=None):
        if z is None:
            # Sample from latent space
            z = self.sample_latent(pc.shape[0])
        z = z.to(self.device).float()
        if not (extra_cond is None):
            extra_cond.to(self.device).float()
        predicted_grasp, confidence = self.decode(pc.to(self.device).float(), z, extra_cond)
        return predicted_grasp, confidence, z.squeeze()

    def generate_dense_latents(self, resolution):
        """
        For the VAE sampler we consider dense latents to correspond to those between -2 and 2
        """
        latents = torch.meshgrid(*[
            torch.linspace(-2, 2, resolution) for i in range(self.latent_size)
        ])
        return torch.stack([latents[i].flatten() for i in range(len(latents))],
                           dim=-1)

# Just a simple MLP
class ScoreFunction(nn.Module):
    def __init__(self, input_dim):
        super(ScoreFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, z):
        return self.pred_score(z).sigmoid()

    def pred_score(self,z):
        z = self.fc1(z).relu()
        z = self.fc2(z).relu()
        return self.out(z)

def base_network(pointnet_radius, pointnet_nclusters, scale, in_features):
    """
    Create network from PointNet
    :param pointnet_radius:
    :param pointnet_nclusters:
    :param scale:
    :param in_features:
    :return:
    """
    sa1_module = pointnet2.PointnetSAModule(
        npoint=pointnet_nclusters, # Number of points in point cloud
        radius=pointnet_radius,
        nsample=64,
        mlp=[in_features, 32 * scale, 32 * scale, 64 * scale])
    sa2_module = pointnet2.PointnetSAModule(
        npoint=32,
        radius=0.04,
        nsample=128,
        mlp=[64 * scale, 64 * scale, 64 * scale, 128 * scale])

    sa3_module = pointnet2.PointnetSAModule(
        mlp=[128 * scale, 128 * scale, 128 * scale, 256 * scale])

    sa_modules = nn.ModuleList([sa1_module, sa2_module, sa3_module])
    fc_layer = nn.Sequential(nn.Linear(256 * scale, 512 * scale),
                             nn.BatchNorm1d(512 * scale), nn.ReLU(True),
                             nn.Linear(512 * scale, 512 * scale),
                             nn.BatchNorm1d(512 * scale), nn.ReLU(True))
    return nn.ModuleList([sa_modules, fc_layer])
