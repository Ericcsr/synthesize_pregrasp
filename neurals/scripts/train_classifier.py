import os
import time
import neurals.dataset
import neurals.dex_grasp_net as dgn
from neurals.network import ScoreFunction, LargeScoreFunction
from neurals.train_options import TrainOptions
import torch.utils.data
import torch
import torch.nn as nn

current_dir = "/home/ericcsr/sirui/contact_planning_dexterous_hand/neurals/pretrained_score_function"

def parse_input(data):
    return data['point_cloud'].cuda().float(), data['fingertip_pos'].cuda().float(), data['label'].cuda().float()

def main():
    parser = TrainOptions()
    # Add loading model related issue
    parser.parser.add_argument(
        '--store_timestamp',
        type=str,
        default=None,
        help='Which timestamp folder to load from'
    )
    parser.parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Name of save folder'
    )
    parser.parser.add_argument(
        '--use_object_model',
        action='store_true'
    )
    parser.parser.add_argument(
        '--use_large_model',
        action='store_true',
        default=False
    )
    opt = parser.parse()
    if opt == None:
        return
    use_wandb = opt.use_wandb
    # TODO: Ask albert regarding hard coded
    objects = ("003_cracker_box",)
    print('objects', objects)

    if use_wandb:
        import wandb
        wandb.init()
        wandb.config.update(opt)
        wandb.config.update({'objects': objects})
    if opt.use_large_model:
        score_function = LargeScoreFunction(num_fingers=3).cuda()
    else:
        model = dgn.DexGraspNetModel(opt, pred_base=False, pred_fingers=[2], extra_cond_fingers=[0,1])
        # Create model for score predictor
        score_function = ScoreFunction(model.get_latent_size()).cuda()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(score_function.parameters(), lr=1e-3)

    # TODO: Make kin_feasibility and dyn_feasibility working
    #full_dataset = neurals.dataset.ScoreFunctionDataset(positive_filepaths=["positive"], negative_filepaths=["negative"])
    full_dataset = neurals.dataset.SmallDataset(positive_grasp_files=["pose_0","pose_1","pose_2","pose_3","pose_4"],
                                                negative_grasp_files=["pose_0","pose_1","pose_2","pose_3","pose_4", "pose_5", "pose_6", "pose_7"],
                                                point_clouds=["pose_0_pcd","pose_1_pcd","pose_2_pcd","pose_3_pcd","pose_4_pcd", "pose_5_pcd", "pose_6_pcd", "pose_7_pcd"])

    # Split the dataset
    dataset_size = len(full_dataset)
    test_size = int(dataset_size*opt.test_ratio)
    train_size = dataset_size - test_size
    print(f"Total size: {dataset_size} Training size: {train_size} Test size: {test_size}")
    (training_dataset, test_dataset) = \
        torch.utils.data.random_split(
            full_dataset, (train_size, test_size))
    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=opt.batch_size)
    # TODO: Ask Michelle why not using batch training?
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    if use_wandb:
        if not opt.use_large_model:
            wandb.config.update({'save_dir': model.save_dir})    
    # writer = Writer(opt)
    total_steps = 0
    print('Dataset loaded, beginning training')
    if not opt.use_large_model:
        model.eval() # The encoder and decoder should not be trained
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        cum_loss = 0
        cum_cnt = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            pcd, tip_pos, label = parse_input(data)
            if not opt.use_large_model:
                grasp_idx = model.finger_to_idx([2])
                cond_idx = model.finger_to_idx([0, 1])
                latent_state = model.encode(pcd, tip_pos[:,grasp_idx], tip_pos[:,cond_idx])
                pred_sigmoid_score = score_function(latent_state).float()
            else:
                pred_sigmoid_score = score_function(pcd, tip_pos).float()
            loss = loss_fn(pred_sigmoid_score, label.view(-1,1))
            loss.backward()
            optimizer.step()
            cum_loss += float(loss)
            cum_cnt += 1
            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.batch_size
                print(f"Time: {t} Cumulative Loss: {cum_loss/cum_cnt}")
                cum_loss = 0
                cum_cnt = 0
        if epoch % 20 == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, total_steps))
            if opt.use_large_model:
                torch.save(score_function.state_dict(), f"{current_dir}/large_model_{epoch}.pth")
            else:
                torch.save(score_function.state_dict(), f"{current_dir}/model_{epoch}.pth")

            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay,
               time.time() - epoch_start_time))
    #     if opt.verbose_plot:
    #         writer.plot_model_wts(model, epoch)
    #
        if epoch % opt.run_test_freq == 0:
            # Turn off training
            if not opt.use_large_model:
                model.eval()
            score_function.eval()
            with torch.no_grad():
                loss_test_all = 0.
                for i, test_data in enumerate(test_loader):
                    pcd, fingertip_pos, label = parse_input(test_data)
                    if not opt.use_large_model:
                        grasp_idx = model.finger_to_idx([2])
                        cond_idx = model.finger_to_idx([0, 1])
                        latent_state = model.encode(pcd, fingertip_pos[:,grasp_idx], fingertip_pos[:,cond_idx])
                        pred_score_sigmoid = score_function(latent_state)
                    else:
                        pred_score_sigmoid = score_function(pcd, fingertip_pos)
                    loss = loss_fn(pred_score_sigmoid, label.view(1,-1))
                    loss_test_all += loss
                loss_test_all /= test_size
                wandb.log({
                    'epoch': epoch,
                    'epoch_iter': epoch_iter,
                    't': t,
                    't_data': t_data,
                    'test_total_loss': loss_test_all
                })
            # Turn training back on
            score_function.train()

    # writer.close()


if __name__ == '__main__':
    main()
