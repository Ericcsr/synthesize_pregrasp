import torch.utils.data
import training.network as network
import training.train_options as to
import training.dex_grasp_net as dgn
import training.dataset
import unittest

class TestDexGraspNet(unittest.TestCase):
    def setUp(self) -> None:
        self.data = training.dataset.make_dataset_from_same_point_cloud('005_tomato_soup_can_test', 'YcbTomatoSoupCan')
        self.opt = to.TrainOptions().parse()
        self.opt.continue_train = False
        self.opt.batch_size = 4
        self.train_model = dgn.DexGraspNetModel(self.opt)

    def test_forward_train(self):
        loader = torch.utils.data.DataLoader(self.data, batch_size=self.opt.batch_size)
        for idx, sample in enumerate(loader):
            self.train_model.set_input(sample)
            predicted_grasp, confidence, mu, logvar = self.train_model.forward()
            self.assertEqual(predicted_grasp.cpu().detach().numpy().shape,
                             (self.opt.batch_size, 23))
            self.assertEqual(confidence.cpu().detach().numpy().shape,
                             (self.opt.batch_size,))
            break

    def test_optimizer(self):
        """
        Do a full iteration of training
        :return:
        """
        loader = torch.utils.data.DataLoader(self.data, batch_size=self.opt.batch_size)
        initial_loss = 0.
        final_loss = 0.
        num_epochs = 3
        for epoch in range(num_epochs):
            for idx, sample in enumerate(loader):
                self.train_model.set_input(sample)
                self.train_model.optimize_parameters()
                if epoch==0:
                    initial_loss += self.train_model.loss_train.cpu().detach().numpy()
                if epoch==num_epochs-1:
                    final_loss += self.train_model.loss_train.cpu().detach().numpy()
        self.assertLess(final_loss, initial_loss)

if __name__ == '__main__':
    unittest.main()