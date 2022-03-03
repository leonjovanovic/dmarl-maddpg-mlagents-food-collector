import torch.cuda
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel import IncomingMessage
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

import Config
import NN

MODEL_NAME = "FoodCollector_28.8.31.25_2.pt"

path = 'models/' + MODEL_NAME

# This is a non-blocking call that only loads the environment.
channel = StatsSideChannel()
msg = IncomingMessage(buffer=bytes(1000))
channel.on_message_received(msg)
env = UnityEnvironment(
    file_name='D:/Users/Leon Jovanovic/Documents/Computer Science/Unity Projects/ml-agents/Project/BuildFoodCollector/UnityEnvironment.exe',
    seed=1, side_channels=[channel])
env.reset()
behaivor_name = next(iter(env.behavior_specs.keys()))
behavior_specs = next(iter(env.behavior_specs.values()))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_shape = behavior_specs[0][0].shape
policy_nn = NN.PolicyNN(state_shape[0] * state_shape[1] * state_shape[2], behavior_specs[1].continuous_size, behavior_specs[1].discrete_branches[0]).to(device)
policy_nn.load_state_dict(torch.load(path))

steps = list(env.get_steps(behaivor_name))
decision_steps = steps[0]
print(decision_steps.obs[0].shape)
while True:
    state_t = torch.flatten(torch.Tensor(decision_steps.obs[0]).to(device), start_dim=1)
    action_cont, action_disc = policy_nn(state_t)
    actionTuple = ActionTuple(continuous=action_cont.detach().cpu().numpy(), discrete=action_disc.detach().cpu().numpy())
    env.set_actions(action=actionTuple, behavior_name=behaivor_name)
    env.step()
    steps = list(env.get_steps(behaivor_name))
    decision_steps = steps[0]
    terminal_steps = steps[1]

# input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(6)]
# output_names = ["output1"]
# torch.onnx.export(policy_nn, torch.Tensor(decision_steps.obs[0]).to(device),MODEL_NAME+".onnx", input_names=input_names, output_names=output_names, verbose=True)
