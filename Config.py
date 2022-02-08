import datetime

total_steps = 100000
# koliko dugo idemo random steps, treba nekih 10k msm
start_steps = 3000
test_every = 5000

buffer_size = 10000
min_buffer_size = 1000
batch_size = 64

num_of_agents = 5
num_of_envs = 4

policy_lr = 0.0003
critic_lr = 0.0004
adam_eps = 1e-8

gamma = 0.99
polyak = 0.995