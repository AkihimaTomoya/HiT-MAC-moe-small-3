from __future__ import division
from setproctitle import setproctitle as ptitle

import os
import glob
import time
import torch
import shutil
import logging
import numpy as np
from tensorboardX import SummaryWriter
import pickle
from model import build_model
from utils import setup_logger
from player_util import Agent
from environment import create_env


def save_checkpoint(args, model, optimizer, episode, n_iter, checkpoint_dir, keep_last=3):
    """
    Lưu checkpoint và logger files
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'episode': episode,
        'n_iter': n_iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'log_dir': args.log_dir
    }
    
    # 1. Lưu checkpoint mới nhất (luôn ghi đè)
    latest_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # 2. Lưu checkpoint theo episode (không ghi đè)
    episode_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.pth')
    torch.save(checkpoint, episode_path)
    
    # 3. Backup logger files
    logger_dir = os.path.join(args.log_dir, 'logger')
    if os.path.exists(logger_dir):
        logger_backup_dir = os.path.join(checkpoint_dir, 'logger_backups')
        if not os.path.exists(logger_backup_dir):
            os.makedirs(logger_backup_dir)
        
        # Copy logger file với tên theo episode
        logger_file = os.path.join(logger_dir, f'{args.env}_log')
        if os.path.exists(logger_file):
            # Backup theo episode
            logger_backup_file = os.path.join(logger_backup_dir, f'logger_ep{episode}.log')
            shutil.copy2(logger_file, logger_backup_file)
            
            # Backup latest
            latest_logger = os.path.join(logger_backup_dir, 'latest_logger.log')
            shutil.copy2(logger_file, latest_logger)
            
            print(f"[Test] Backed up logger: logger_ep{episode}.log")
    
    # 4. Xóa checkpoint cũ
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_ep*.pth'))
    
    def get_episode_number(path):
        try:
            return int(path.split('_ep')[-1].split('.pth')[0])
        except:
            return 0
    
    checkpoints.sort(key=get_episode_number, reverse=True)
    
    # Xóa các checkpoint và logger cũ
    logger_backup_dir = os.path.join(checkpoint_dir, 'logger_backups')
    for old_checkpoint in checkpoints[keep_last:]:
        try:
            ep_num = get_episode_number(old_checkpoint)
            os.remove(old_checkpoint)
            print(f"[Test] Removed old checkpoint: {os.path.basename(old_checkpoint)}")
            
            # Xóa logger backup tương ứng
            old_logger = os.path.join(logger_backup_dir, f'logger_ep{ep_num}.log')
            if os.path.exists(old_logger):
                os.remove(old_logger)
                print(f"[Test] Removed old logger: logger_ep{ep_num}.log")
        except Exception as e:
            print(f"[Test] Failed to remove {old_checkpoint}: {e}")
    
    print(f"[Test] Checkpoint saved at episode {episode}, iteration {n_iter}")
    print(f"       Kept {min(len(checkpoints), keep_last)} checkpoint(s)")


def test(args, shared_model, optimizer, train_modes, n_iters, episode_counter=None, 
         checkpoint_dir=None, save_interval=100, keep_checkpoints=3):
    ptitle('Test Agent')
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Test'))
    gpu_id = args.gpu_ids[-1]
    log = {}
    
    # Setup logger với append mode nếu resume
    logger_dir = os.path.join(args.log_dir, 'logger')
    logger_file = os.path.join(logger_dir, f'{args.env}_log')
    
    # Kiểm tra nếu đang resume và logger file đã tồn tại
    if args.resume and os.path.exists(logger_file):
        # Logger file đã được restore từ checkpoint, sử dụng append mode
        setup_logger('{}_log'.format(args.env),
                     logger_file,
                     mode='a')  # Append mode
        print(f"[Test] Continuing existing logger at: {logger_file}")
    else:
        # Tạo logger mới với write mode
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        setup_logger('{}_log'.format(args.env),
                     logger_file,
                     mode='w')  # Write mode
        print(f"[Test] Created new logger at: {logger_file}")
    
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(args.env))
    
    # Log args
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    env = create_env(args.env, args)
    env.seed(args.seed)
    start_time = time.time()
    count_eps = 0

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(player.env.observation_space, player.env.action_space, args, device).to(device)
    player.model.eval()
    max_score = -100
    max_avg_score = -100
    
    # Load episode counter từ shared value
    episode = episode_counter.value if episode_counter else 0
    initial_episode = episode
    
    print(f"\n[Test] Starting test loop from episode {episode}")
    
    avg_reward_eps = 0
    result = []
    
    while True:
        AG = 0
        reward_sum = np.zeros(player.num_agents)
        reward_sum_list = []
        len_sum = 0
        
        for i_episode in range(args.test_eps):
            player.model.load_state_dict(shared_model.state_dict())
            player.reset()
            reward_sum_ep = np.zeros(player.num_agents)
            rotation_sum_ep = 0

            fps_counter = 0
            t0 = time.time()
            count_eps += 1
            fps_all = []
            
            while True:
                player.action_test()
                fps_counter += 1
                reward_sum_ep += player.reward
                rotation_sum_ep += player.rotation
                
                if player.done:
                    AG += reward_sum_ep[0]/rotation_sum_ep*player.num_agents
                    reward_sum += reward_sum_ep
                    reward_sum_list.append(reward_sum_ep[0])
                    len_sum += player.eps_len
                    fps = fps_counter / (time.time()-t0)
                    n_iter = 0
                    for n in n_iters:
                        n_iter += n

                    for i, r_i in enumerate(reward_sum_ep):
                        writer.add_scalar('test/reward'+str(i), r_i, n_iter)

                    fps_all.append(fps)
                    writer.add_scalar('test/fps', fps, n_iter)
                    writer.add_scalar('test/eps_len', player.eps_len, n_iter)
                    break

        # Calculate averages
        ave_AG = AG/args.test_eps
        ave_reward_sum = reward_sum/args.test_eps
        len_mean = len_sum/args.test_eps
        reward_step = reward_sum / len_sum
        mean_reward = np.mean(reward_sum_list)
        std_reward = np.std(reward_sum_list)
        
        # Tăng episode counter
        episode += 1
        
        # Cập nhật shared episode counter
        if episode_counter:
            episode_counter.value = episode
        
        log['{}_log'.format(args.env)].info(
            "Time {0}, Episode {1}, ave eps reward {2}, ave eps length {3}, reward step {4}, FPS {5}, "
            "mean reward {6}, std reward {7}, AG {8}".
            format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                episode,
                np.around(ave_reward_sum, decimals=2), np.around(len_mean, decimals=2),
                np.around(reward_step, decimals=2), np.around(np.mean(fps_all), decimals=2),
                mean_reward, std_reward, np.around(ave_AG, decimals=2)
            ))
        
        avg_reward_eps += ave_reward_sum[0]
        result.append(avg_reward_eps/episode)
        
        # Lưu checkpoint theo interval
        if checkpoint_dir and (episode - initial_episode) % save_interval == 0 and episode > initial_episode:
            save_checkpoint(args, shared_model, optimizer, episode, n_iter, checkpoint_dir, keep_checkpoints)
            print(f"[Test] Regular checkpoint saved (interval: {save_interval})")
        
        # Save model
        save_model = False
        if ave_reward_sum[0] >= max_score:
            print(f'[Test] New best score: {max_score:.5f} -> {ave_reward_sum[0]:.5f}')
            max_score = ave_reward_sum[0]
            model_dir = os.path.join(args.log_dir, 'best.pth')
            save_model = True
        elif np.mean(ave_reward_sum) >= max_avg_score:
            print(f'[Test] New best avg score: {max_avg_score:.5f} -> {np.mean(ave_reward_sum):.5f}')
            max_avg_score = np.mean(ave_reward_sum)
            model_dir = os.path.join(args.log_dir, 'best_avg.pth')
            save_model = True
        else:
            model_dir = os.path.join(args.log_dir, 'new.pth'.format(args.env))
        
        # Lưu checkpoint khi có best model
        if save_model and checkpoint_dir:
            save_checkpoint(args, shared_model, optimizer, episode, n_iter, checkpoint_dir, keep_checkpoints)
            print(f"[Test] Best model checkpoint saved at episode {episode}")
        
        log['{}_log'.format(args.env)].info(
            "Episode {0} - BEST: {1} | BEST OVERALL: {2} | AVG: {3} | Iter: {4}".
            format(
                episode, np.around(max_score, decimals=5), np.around(max_avg_score, decimals=5),
                np.around(avg_reward_eps/episode, decimals=5), n_iter
            ))
        
        state_to_save = {"model": player.model.state_dict(),
                         "optimizer": optimizer.state_dict() if optimizer else None}
        torch.save(state_to_save, model_dir)
        time.sleep(args.sleep_time)
        
        if n_iter > args.max_step:
            # Lưu final checkpoint trước khi thoát
            if checkpoint_dir:
                save_checkpoint(args, shared_model, optimizer, episode, n_iter, checkpoint_dir, keep_checkpoints)
                print("[Test] Final checkpoint saved!")
            
            env.close()
            for id in range(0, args.workers):
                train_modes[id] = -100
            break
