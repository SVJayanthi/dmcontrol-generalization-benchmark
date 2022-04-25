import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from logger import Logger
from video import VideoRecorder
from torchvision import transforms
import augmentations
from pathlib import Path

def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode='train'
    )
    test_env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed+42,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.eval_mode,
        intensity=args.distracting_cs_intensity
    ) if args.eval_mode is not None else None

    # Create working directory
    work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed))
    print('Working directory:', work_dir)
    assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    utils.write_info(args, os.path.join(work_dir, 'info.log'))

    assert torch.cuda.is_available(), 'must have cuda enabled'
    cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
    print('Observations:', env.observation_space.shape)
    print('Cropped observations:', cropped_obs_shape)

    start_step, episode, episode_reward, done = 0, 0, 0, True
    L = Logger(work_dir)
    start_time = time.time()
    rewards = []

    dataset_dir = '/home/taylor/Desktop/rlr/dmcontrol_data'
    num_domains = 5
    saved_obs = list([[] for i in range(num_domains)])

    for step in range(start_step, args.train_steps+1):
        if done:
            if step > start_step:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        # Sample action for data collection
        action = env.action_space.sample()

        if np.random.rand() < 0.1: # TODO: maybe weight by how far along episode is
            dim_added_obs = obs._force()[np.newaxis, ...]
            class_num = np.random.choice(range(num_domains))
            saved_obs[class_num].append(dim_added_obs)

        # Save batch of observations
        for class_num in range(num_domains):
            if len(saved_obs[class_num]) >= 1000:
                batch_saved_obs = np.concatenate(saved_obs[class_num], axis=0)

                class_dir = f'dmcontrol{class_num + 1}'
                if class_num == 1:
                    batch_saved_obs = utils.transform_batch_obs(batch_saved_obs, transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=[-7/18, -7/18]))
                elif class_num == 2:
                    batch_saved_obs = utils.transform_batch_obs(batch_saved_obs, transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=[-3/18, -3/18]))
                elif class_num == 3:
                    batch_saved_obs = utils.transform_batch_obs(batch_saved_obs, transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=[3/18, 3/18]))
                elif class_num == 4:
                    batch_saved_obs = utils.transform_batch_obs(batch_saved_obs, transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=[7/18, 7/18]))

                batch_saved_obs = np.transpose(batch_saved_obs, (0, 2, 3, 1))

                file_index = 0
                while os.path.exists(os.path.join(dataset_dir, class_dir, f'{file_index}.npz')):
                    file_index += 1
                dir = os.path.join(dataset_dir, class_dir)
                path = os.path.join(dir, f'{file_index}.npz')
                print(f'Saving {len(saved_obs[class_num])} observations to {path}')

                Path(dir).mkdir(parents=True, exist_ok=True)
                np.savez(path, obs=batch_saved_obs)
                saved_obs[class_num] = []

        # Take step
        next_obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

    print('Completed gathering observations for', work_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
