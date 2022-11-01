import training.dataset
import unittest
import model.rigid_body_model as rbm
import model.param as model_param
import numpy as np
from scipy.spatial.transform import Rotation as R

class TestDexGraspDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.hand_plant = rbm.AllegroHandPlantDrake(meshcat_open_brower=False,
                                           num_viz_spheres=0)

    def test_make_dataset_from_same_point_cloud(self):
        rs = np.random.RandomState(0)
        copies = 3
        dut = training.dataset.make_dataset_from_same_point_cloud('003_cracker_box',
                                                                make_datset=False, random_xy=False,
                                                                random_yaw=False,
                                                                copies=copies,
                                                                data_idx_start=0, data_idx_end=1,
                                                                random_state=rs)
        # Theres should be exactly 10 data points
        self.assertEqual(len(dut), copies)
        # construct with random orientation
        dut_orn = training.dataset.make_dataset_from_same_point_cloud('003_cracker_box',
                                                                make_datset=False, random_xy=False,
                                                                random_yaw=True,
                                                                copies=copies,
                                                                data_idx_start=0, data_idx_end=1,
                                                                random_state=rs)

        def test_dataset_consistency(d):
            # Check that fingertip_normals is consistent with the fingertip positions
            # computed from (base_position, base_quaternion, finger_q)
            diagram_context, plant_context = self.hand_plant.create_context()
            gt_fingertip_angles_dict = {}
            gt_fingertip_angles_dict = model_param.finger_q_to_finger_angles_dict(d.finger_q)
            gt_fingertip_angles_dict[model_param.AllegroHandFinger.RING] = np.zeros(4)
            # Compute the corresponding fingeritp locations
            gt_drake_q = self.hand_plant.convert_hand_configuration_to_q(d.base_position,
                d.base_quaternion, gt_fingertip_angles_dict
            )
            gt_p_WF = self.hand_plant.compute_p_WF(gt_drake_q, plant_context)
            gt_fingertip_normals = d.fingertip_normals
            for fi, finger in enumerate(model_param.ActiveAllegroHandFingers):
                np.testing.assert_allclose(gt_fingertip_normals[fi,:3], 
                np.squeeze(gt_p_WF[finger]),
                atol=1e-2 # Allow large tolerance as gt_p_WF is from IK solution
            )

        for idx in range(len(dut)):
            # Retrieve the random orientation
            original_entry = dut[idx]
            rotated_entry = dut_orn[idx]
            original_base_quaternion = R.from_quat(original_entry.base_quaternion[[1,2,3,0]])
            rotated_base_quaternion = R.from_quat(rotated_entry.base_quaternion[[1,2,3,0]])
            rot_diff = rotated_base_quaternion*original_base_quaternion.inv()
            # Rotate everything else back
            rot_diff_matrix = rot_diff.as_matrix()
            # Compare point clouds
            pc_original = original_entry.point_cloud
            pc_rot = rotated_entry.point_cloud
            np.testing.assert_allclose(pc_rot, 
                                        (rot_diff_matrix @ (pc_original.T)).T)
            # Compare normals
            fingertip_normals_original = original_entry.fingertip_normals
            fingertip_normals_rot = rotated_entry.fingertip_normals
            # Check fingertip positions
            np.testing.assert_allclose(fingertip_normals_rot[:,:3], 
                                        (rot_diff_matrix @ (fingertip_normals_original[:,:3].T)).T)
            # Check fingertip normal vectors
            np.testing.assert_allclose(fingertip_normals_rot[:,3:], 
                            (rot_diff_matrix @ (fingertip_normals_original[:,3:].T)).T)
            # Check finger q
            np.testing.assert_allclose(original_entry.finger_q, rotated_entry.finger_q)

            # Check base position
            np.testing.assert_allclose(rotated_entry.base_position, rot_diff_matrix @ original_entry.base_position)

            # Test consistency
            test_dataset_consistency(original_entry)
            test_dataset_consistency(rotated_entry)

        # construct with random translation
        dut_trans = training.dataset.make_dataset_from_same_point_cloud('003_cracker_box',
                                                                make_datset=False, random_xy=True,
                                                                random_yaw=False,
                                                                copies=copies,
                                                                data_idx_start=0, data_idx_end=1,
                                                                random_state=rs)
        for idx in range(len(dut)):
            # Retrieve the random translation
            original_entry = dut[idx]
            translated_entry = dut_trans[idx]
            trans = translated_entry.base_position-original_entry.base_position
            # Compare point clouds
            pc_original = original_entry.point_cloud
            pc_trans = translated_entry.point_cloud
            pc_diff = pc_trans-pc_original
            self.assertTrue(np.allclose(pc_diff, trans))
            # Compare normals
            fingertip_normals_original = original_entry.fingertip_normals
            fingertip_normals_trans = translated_entry.fingertip_normals
            # Check fingertip positions
            self.assertTrue(np.allclose((fingertip_normals_trans-fingertip_normals_original)[:,:3], 
                                        trans))
            # Check fingertip normal vectors
            np.testing.assert_allclose(fingertip_normals_trans[:,3:], fingertip_normals_original[:,3:])
            # Check finger q
            np.testing.assert_allclose(original_entry.finger_q, translated_entry.finger_q)
            np.testing.assert_allclose(original_entry.base_quaternion, translated_entry.base_quaternion)
            test_dataset_consistency(original_entry)
            test_dataset_consistency(translated_entry)

if __name__ == '__main__':
    unittest.main()