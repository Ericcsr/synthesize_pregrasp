import torch
import math
import pytorch_kinematics as pk

chain = pk.build_serial_chain_from_urdf(open("simox_ros\\sr_grasp_description\\urdf\\shadowhand.urdf").read(), "fftip")
chain2 = pk.build_serial_chain_from_urdf(open("simox_ros\\sr_grasp_description\\urdf\\shadowhand.urdf").read(), "mftip")

# require gradient through the input joint values
th = torch.tensor([0.0] * 29, requires_grad=True).double()

tg1 = chain.forward_kinematics(th)
#tg2 = chain2.forward_kinematics(th)
m = tg1.get_matrix()
#m2 = tg2.get_matrix()
pos =   m[:, :3, 3]
#pos2 = m2[:, :3, 3]
loss = pos.norm()# + pos2.norm()
loss.backward(retain_graph=True)
th.retain_grad()
# now th.grad is populated
print(th.grad)
# -0.680827560572996, 2.5165493061814204e-12, 0.37414213562384196
# -0.4673, -0.4673,  0.3741
# (-0.6512453454775682, -0.6512453454680527, 0.36000000000014826)
# -0.9210000000000002, 6.728596412680643e-12, 0.36000000000014826

# Need to solve a optimization problem to ensure contact.