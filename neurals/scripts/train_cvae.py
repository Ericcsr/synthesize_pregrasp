import time
import neurals.dataset
import neurals.dex_grasp_net as dgn
from neurals.train_options import TrainOptions
import torch.utils.data

def main():
    opt = TrainOptions().parse()
    if opt == None:
        return
    use_wandb = opt.use_wandb

    if use_wandb:
        import wandb
        wandb.init()
        wandb.config.update(opt)
    model = dgn.DexGraspNetModel(opt, pred_base=False,pred_fingers=[2,3,4], extra_cond_fingers=[0,1], gpu_id=0)
    full_dataset = neurals.dataset.SmallDataset(seed_folder="seeds_scale",
                                                point_clouds=["pose_00_pcd","pose_01_pcd","pose_02_pcd","pose_03_pcd",
                                                              "pose_04_pcd","pose_05_pcd","pose_06_pcd","pose_07_pcd",
                                                              "pose_08_pcd","pose_09_pcd","pose_10_pcd","pose_11_pcd","pose_12_pcd",
                                                              "pose_13_pcd","pose_14_pcd","pose_15_pcd","pose_16_pcd",
                                                              "pose_17_pcd","pose_18_pcd","pose_19_pcd","pose_20_pcd",
                                                              "pose_21_pcd","pose_22_pcd","pose_23_pcd","pose_24_pcd",
                                                              "pose_25_pcd","pose_26_pcd","pose_27_pcd","pose_28_pcd","pose_29_pcd",
                                                              "pose_30_pcd","pose_31_pcd","pose_32_pcd"])

    # Split the dataset
    dataset_size = len(full_dataset)
    test_size = int(dataset_size*opt.test_ratio)
    train_size = dataset_size - test_size
    (training_dataset, test_dataset) = \
        torch.utils.data.random_split(
            full_dataset, (train_size, test_size))
    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=opt.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    if use_wandb:
        wandb.config.update({'save_dir': model.save_dir})
    # writer = Writer(opt)
    total_steps = 0
    print('Dataset loaded, beginning training')
    model.train()
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.print_freq == 0:
                loss_types = []
                if opt.arch == "vae":
                    # For vae, reconstruction loss is just the L1 loss
                    loss = [
                        model.loss_train, model.kl_loss_train, model.fingertip_pos_loss_train
                    ]
                    loss_types = [
                        "total_loss", "kl_loss", "fingertip_pos_loss"
                    ]
                else:
                    loss = [
                        model.loss_train, model.classification_loss,
                    ]
                    loss_types = [
                        "total_loss", "classification_loss"
                    ]
                t = (time.time() - iter_start_time) / opt.batch_size
                if use_wandb:
                    log_dict = {'epoch': epoch,
                                'epoch_iter': epoch_iter,
                                't': t,
                                't_data': t_data
                                }
                    for l_idx, lt in enumerate(loss_types):
                        log_dict[lt] = loss[l_idx]
                    wandb.log(log_dict)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest', epoch)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest', epoch)
            model.save_network(str(epoch), epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay,
               time.time() - epoch_start_time))
        model.update_learning_rate()
    #     if opt.verbose_plot:
    #         writer.plot_model_wts(model, epoch)
    #
        if epoch % opt.run_test_freq == 0:
            # Turn off training
            model.eval()
            with torch.no_grad():
                loss_test_all = 0.
                kl_loss_test_all = 0.
                fingertip_pos_loss_test_all = 0.
                for i, test_data in enumerate(test_loader):
                    model.set_input_test(test_data)
                    model.compute_loss_on_test_data()
                    loss_test_all += model.loss_test
                    kl_loss_test_all += model.kl_loss_test
                    fingertip_pos_loss_test_all += model.fingertip_pos_loss_test
                loss_test_all /= test_size
                kl_loss_test_all /= test_size
                fingertip_pos_loss_test_all /= test_size
                wandb.log({
                    'epoch': epoch,
                    'epoch_iter': epoch_iter,
                    't': t,
                    't_data': t_data,
                    'test_total_loss': loss_test_all,
                    'test_kl_loss': kl_loss_test_all,
                    'test_fingertip_pos_loss': fingertip_pos_loss_test_all
                })
            # Turn training back on
            model.train()

    # writer.close()


if __name__ == '__main__':
    main()
