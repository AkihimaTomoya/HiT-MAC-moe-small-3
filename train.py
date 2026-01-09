from __future__ import division
import os
import time
import torch
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from setproctitle import setproctitle as ptitle

from model import build_model
from player_util import Agent
from environment import create_env


def train(rank, args, shared_model, optimizer, train_modes, n_iters, episode_counter=None, 
          start_n_iter=0, env=None):
    n_iter = start_n_iter  # Bắt đầu từ iteration đã lưu
    writer = SummaryWriter(os.path.join(args.log_dir, 'Agent:{}'.format(rank)))
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    training_mode = args.train_mode
    env_name = args.env

    train_modes.append(training_mode)
    n_iters.append(n_iter)

    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
        device = torch.device('cuda:' + str(gpu_id))
        if len(args.gpu_ids) > 1:
            device_share = torch.device('cpu')
        else:
            device_share = torch.device('cuda:' + str(args.gpu_ids[-1]))
    else:
        device = device_share = torch.device('cpu')
    
    if env == None:
        env = create_env(env_name, args, rank)

    params = shared_model.parameters()
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=args.lr)
    
    if args.fix:
        env.seed(args.seed)
    else:
        env.seed(rank % (args.seed + 1))
    
    player = Agent(None, env, args, None, device)
    player.rank = rank
    player.gpu_id = gpu_id

    # Prepare model
    player.model = build_model(player.env.observation_space, player.env.action_space, args, device)
    player.model = player.model.to(device)
    player.model.train()

    player.reset()
    reward_sum = torch.zeros(player.num_agents).to(device)
    reward_sum_org = np.zeros(player.num_agents)
    ave_reward = np.zeros(2)
    ave_reward_longterm = np.zeros(2)
    count_eps = 0
    
    # Log resume info
    if start_n_iter > 0:
        print(f"[Rank {rank}] Resuming training from iteration {start_n_iter}")
    
    while True:
        # Sync to the shared model
        player.model.load_state_dict(shared_model.state_dict())

        if player.done:
            player.reset()
            reward_sum = torch.zeros(player.num_agents).to(device)
            reward_sum_org = np.zeros(player.num_agents)
            count_eps += 1

        player.update_rnn_hidden()
        t0 = time.time()
        
        for s_i in range(args.num_steps):
            player.action_train()
            reward_sum += player.reward
            reward_sum_org += player.reward_org

            if player.done:
                for i, r_i in enumerate(reward_sum):
                    writer.add_scalar('train/reward_' + str(i), r_i, player.n_steps)
                    if args.norm_reward:
                        writer.add_scalar('train/reward_org_' + str(i), reward_sum_org[i].sum(), player.n_steps)
                break
        
        fps = s_i / (time.time() - t0)

        policy_loss, value_loss, entropies = player.optimize(params, optimizer, shared_model, training_mode,
                                                             device_share)

        writer.add_scalar('train/policy_loss_sum', policy_loss.sum(), player.n_steps)
        writer.add_scalar('train/value_loss_sum', value_loss.sum(), player.n_steps)
        writer.add_scalar('train/entropies_sum', entropies.sum(), player.n_steps)

        writer.add_scalar('train/ave_reward', ave_reward[0] - ave_reward_longterm[0], player.n_steps)
        writer.add_scalar('train/mode', training_mode, player.n_steps)
        writer.add_scalar('train/fps', fps, player.n_steps)

        n_iter += 1
        n_iters[rank] = n_iter
        
        # Log progress mỗi 100 iterations
        if n_iter % 100 == 0:
            print(f"[Rank {rank}] Iteration: {n_iter}, Episode: {count_eps}, FPS: {fps:.2f}")

        if train_modes[rank] == -100:
            env.close()
            break
