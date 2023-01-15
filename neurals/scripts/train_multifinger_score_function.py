import os
import time
import neurals.dataset
import neurals.dex_grasp_net as dgn
from neurals.network import MultifingerScoreFunction
from argparse import ArgumentParser
import torch.utils.data
import torch
import torch.nn as nn

TEST_RATIO = 0.1

current_dir = "neurals/pretrained_score_function"

MASKS = [[0,1,2,3,4,5], [0,1,2], [3,4,5]]
FINGERS = [[0,1], [0], [1]]

def parse_input(data):
    pcd = data['point_cloud'].cuda().float()
    cond = data['condition'].cuda().float()
    score = data['score'].cuda().float()
    pcd_df = None
    if "point_cloud_df" in data.keys():
        pcd_df = data['point_cloud_df'].cuda().float()
        pcd_df = pcd_df.view(pcd_df.shape[0], pcd_df.shape[1], 1)
    return pcd, cond, score, pcd_df

def load_dataset(opt):
    full_dataset = neurals.dataset.ScoreDataset(score_file=opt.dataset_file, noise_scale=0.005, has_distance_field=opt.has_distance_field)
    # Split the dataset
    dataset_size = len(full_dataset)
    test_size = int(dataset_size*TEST_RATIO)
    train_size = dataset_size - test_size
    #print(f"Total size: {dataset_size} Training size: {train_size} Test size: {test_size}")
    (training_dataset, test_dataset) = \
        torch.utils.data.random_split(
            full_dataset, (train_size, test_size))
    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    return train_loader, test_loader, train_size, test_size
    

def main():
    parser = ArgumentParser()
    # Add loading model related issue
    parser.add_argument("--exp_name", type=str, default=None, required=True)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--pretrain_pcn", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--has_distance_field", action="store_true", default=False)
    parser.add_argument("--dataset_file", type=str, default=None, required=True)
    parser.add_argument("--epochs", type=int, default=3000)
    opt = parser.parse_args()

    if opt == None:
        return
    use_wandb = opt.use_wandb
    # TODO: Ask albert regarding hard coded

    if use_wandb:
        import wandb
        wandb.init()
        wandb.config.update(opt)
    score_function = MultifingerScoreFunction(num_fingers=2, latent_dim=opt.latent_dim, has_distance_field=opt.has_distance_field)
    if not (opt.pretrain_pcn is None):
        print("Pretrained PCN is loaded!!!")
        score_function.pcn.load_state_dict(torch.load(f"neurals/pcn_model/{opt.pretrain_pcn}.pth"))
    score_function.cuda()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam([{"params":score_function.fc1.parameters(), "lr":5e-4},
                                  {"params":score_function.fc2.parameters(), "lr":5e-4},
                                  {"params":score_function.out.parameters(), "lr":5e-4}])
    #optimizer = torch.optim.Adam(score_function.parameters(), lr=1e-3)
    total_steps = 0
    print('Dataset loaded, beginning training')
    # Prepare the score saving directory
    os.makedirs(current_dir+f"/only_score_model_{opt.exp_name}", exist_ok=True)

    for epoch in range(opt.epochs):
        # Need to reset dataset periteration to prevent overfit
        train_loader, test_loader, train_size, test_size = load_dataset(opt)
        # dataset preparation end

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        cum_loss = 0
        cum_cnt = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            iter_start_time = time.time()
            if total_steps % 1000 == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += 16
            epoch_iter += 16
            pcd, condition, score, dist_field = parse_input(data)
            pred_score = score_function(pcd, condition[:,MASKS[epoch%3]], dist_field, grasp_idx=FINGERS[epoch%3])
            loss = loss_fn(pred_score, score.view(-1,1))
            loss.backward()
            optimizer.step()
            cum_loss += float(loss)
            cum_cnt += 1
            if total_steps % 1000 == 0:
                t = (time.time() - iter_start_time) / 16
                print(f"Time: {t} Cumulative Loss: {cum_loss/cum_cnt}")
                cum_loss = 0
                cum_cnt = 0
        if epoch % 20 == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, total_steps))
            torch.save(score_function.state_dict(), f"{current_dir}/only_score_model_{opt.exp_name}/{epoch}.pth")
            iter_data_time = time.time()

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.epochs,
                time.time() - epoch_start_time))
        
        if epoch % 5 == 0:
            # Turn off training
            score_function.eval()
            with torch.no_grad():
                loss_test_all = 0.
                for i, test_data in enumerate(test_loader):
                    pcd, fingertip_pos, label, df = parse_input(test_data)
                    pred_score = score_function(pcd, fingertip_pos, df)
                    loss = loss_fn(pred_score, label.view(1,-1))
                    loss_test_all += loss
                loss_test_all /= test_size
                wandb.log({
                    'epoch': epoch,
                    'epoch_iter': epoch_iter,
                    't_data': t_data,
                    'test_total_loss': loss_test_all
                })
            # Turn training back on
            score_function.train()

if __name__ == '__main__':
    main()
