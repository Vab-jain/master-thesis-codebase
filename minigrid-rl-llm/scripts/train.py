import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys
import torch

import utils
from utils import device
from model import ACModel
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from utils import TextDesciptionWrapper, LLMSuggestedMissionWrapper                                                                                                 

from torch_ac.utils.penv import ParallelEnv
# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
# Parameters for validation
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--episodes", type=int, default=50,
                    help="number of episodes of evaluation (default: 50)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--rgb", action="store_true", default=False,
                    help="Observation contains RGB Image")
parser.add_argument("--llm", action="store_true", default=False,
                    help="Observation contains LLM suggested goals")


if __name__ == "__main__":
    args = parser.parse_args()

    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")
    
    # empty the GPU cache before the experiment runs
    if args.llm:
        torch.cuda.empty_cache()

    # Load environments

    envs = []
    wrappers = [FullyObsWrapper,]
    if args.rgb:
        wrappers.extend([RGBImgObsWrapper,])
    if args.llm:
        wrappers.extend([TextDesciptionWrapper, LLMSuggestedMissionWrapper])
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i, wrappers=wrappers))
    txt_logger.info("Environments loaded")
    txt_logger.info("Wrappers: {}\n".format([wrapper.class_name() for wrapper in wrappers]))
    
    env = ParallelEnv(envs)
    

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text, args.rgb)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Saving status and Validation

        if args.save_interval > 0 and update % args.save_interval == 0:
            # Save status
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
            
            
            
            # Validation results (whenever we are saving)
            txt_logger.info("VALIDATION (after every save)")
            agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                                argmax=args.argmax, num_envs=args.procs,
                                use_memory=args.mem, use_text=args.text, use_rgb=args.rgb)
            
            # Initialize logs
            val_logs = {"num_frames_per_episode": [], "return_per_episode": []}

            # Run agent
            val_start_time = time.time()
            
            obss = env.reset()
            
            val_log_done_counter = 0
            val_log_success_counter = 0
            val_log_episode_return = torch.zeros(args.procs, device=device)
            val_log_episode_num_frames = torch.zeros(args.procs, device=device)

            while val_log_done_counter < args.episodes:
                actions = agent.get_actions(obss)
                obss, rewards, terminateds, truncateds, _ = env.step(actions)
                dones = tuple(a | b for a, b in zip(terminateds, truncateds))
                agent.analyze_feedbacks(rewards, dones)

                val_log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
                val_log_episode_num_frames += torch.ones(args.procs, device=device)

                for i, done in enumerate(dones):
                    if done:
                        val_log_done_counter += 1
                        if val_log_episode_return[i].item():
                            val_log_success_counter += 1
                        val_logs["return_per_episode"].append(val_log_episode_return[i].item())
                        val_logs["num_frames_per_episode"].append(val_log_episode_num_frames[i].item())

                mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
                val_log_episode_return *= mask
                val_log_episode_num_frames *= mask

            val_end_time = time.time()
            
            # Print logs
            
            val_num_frames = sum(val_logs["num_frames_per_episode"])
            val_fps = val_num_frames / (val_end_time - val_start_time)
            val_duration = int(val_end_time - val_start_time)
            val_return_per_episode = utils.synthesize(val_logs["return_per_episode"])
            val_num_frames_per_episode = utils.synthesize(val_logs["num_frames_per_episode"])
            val_percent_success = val_log_success_counter/val_log_done_counter
            
            txt_logger.info("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | Win {:.2f}"
                .format(val_num_frames, val_fps, val_duration,
                        *val_return_per_episode.values(),
                        *val_num_frames_per_episode.values(),
                        val_percent_success))
            
