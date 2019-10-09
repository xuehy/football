# Read dump files of football games and store them into trajectories
# AUTHOR: Hongyang Xue
# DATE: 2019-10-09
import six.moves.cPickle as Pickle
from gfootball.env.football_action_set import action_set_dict
from gfootball.env.wrappers import Simple115StateWrapper
import gfootball.env as football_env
import os


env = football_env.create_environment(
    "11_vs_11_easy_stochastic",
    representation='simple115',
    render=False)
default_actions = action_set_dict['default']
simple115wrapper = Simple115StateWrapper(env)

def process_single_dump(filename):
    traj = []
    f = Pickle.load(open(filename, 'rb'))
    for frame in f:
        # frame is a dict with keys {debug,
        # observartion, reward, 'cumulative_reward'}
        act_frame = frame['debug']['action'][0]
        obs_frame = frame['observation']
        if act_frame not in default_actions:
            act_frame = 0
        else:
            act_frame = default_actions.index(act_frame)
        active = obs_frame['left_agent_controlled_player']
        del obs_frame['left_agent_controlled_player']
        obs_frame['active'] = active[0]
        obs_frame = simple115wrapper.observation([obs_frame])
        traj.append([obs_frame.flatten(), act_frame])
    return traj


def process_dumps(path):
    trajectories = []
    for f in os.listdir(path):
        if f.startswith('score') and f.endswith('.dump'):
            traj = process_single_dump(os.path.join(path, f))
            trajectories.append(traj)
    return trajectories

trajs = process_dumps('/home/xuehy/traces/')
print(len(trajs))
