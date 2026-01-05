# Deal or No Deal Negotiation RL Environment

Gym-compatible environment for the Deal or No Deal negotiation task (Lewis et al., 2017; Kwon et al., 2021), following the setting described in 2303.00001v1. This is a multi-issue bargaining environment using coarse dialogue acts.

## Installation

Recommended: create a virtual environment. Then install requirements:

```bash
pip install -r requirements.txt
```

## Usage (Negotiation env)

```python
from deal_or_no_deal_env import register_deal_or_no_deal
from deal_or_no_deal_env.env import NegotiationConfig
import gymnasium as gym

cfg = NegotiationConfig(
    use_dataset=True,
    dataset_script_path="./deal_or_no_dialog_main/deal_or_no_dialog.py",
    dataset_config_name="dialogues",
    max_turns=10,
)
env_id = register_deal_or_no_deal()
env = gym.make(env_id, config=cfg)
obs, info = env.reset()
done = False
while not done:
    # Action is a dict: {"act_type": Discrete(5), "oA": MultiDiscrete([0..counts])}
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
```

Actions (coarse dialogue acts):
- 0: propose(oA)
- 1: insist(oA)
- 2: agree
- 3: disagree
- 4: end

Observation includes:
- `counts` (3, MultiDiscrete): total item counts (books, hats, balls)
- `my_utilities` (3, MultiDiscrete): uA ∈ {0..10}^3
- `partner_utilities` (3, MultiDiscrete): optional, shown if `reveal_partner_utilities=True`
- `last_partner_act` (Discrete(5))
- `last_partner_offer_for_me` (3, MultiDiscrete): most recent oA proposed by partner
- `turns_remaining` (Discrete)

## Environment specification

### Description
Agents A and B negotiate over 3 item types (books, hats, balls). Contexts provide item counts i ∈ {1..4}^3 and private utilities uA,uB ∈ {0..10}^3. Agents alternate turns for up to T steps using coarse dialogue acts: propose(o), insist(o), agree, disagree, end. If both agree on a consistent allocation oA + oB = i, rewards are rA = uA·oA, rB = uB·oB; otherwise 0.

### Observation space
Dict with keys:
- `counts`: MultiDiscrete([5,5,5]) (0..4 allowed for flexibility)
- `my_utilities`: MultiDiscrete([11,11,11])
- `partner_utilities`: MultiDiscrete([11,11,11]) or masked
- `last_partner_act`: Discrete(5)
- `last_partner_offer_for_me`: MultiDiscrete([5,5,5])
- `turns_remaining`: Discrete(T+1)

### Action space
Dict with keys:
- `act_type`: Discrete(5) as above
- `oA`: MultiDiscrete([0..counts]) for propose/insist (ignored for other acts)

Action masking is provided in `info["action_mask"]` as an int8 array with 1 indicating a valid action.

### Rewards
- If agree on valid allocation: rA = uA·oA
- Otherwise: 0

### Termination
- On `agree` to a valid allocation
- On `end`
- When `turns_remaining == 0`

### Episode dynamics
- Random or configurable starting agent
- Simple heuristic partner policy for simulation; plug in your own to self-play or curriculum

### Dataset integration
Set `NegotiationConfig.use_dataset=True` and provide `dataset_script_path` to `deal_or_no_dialog_main/deal_or_no_dialog.py`. Use `dataset_config_name="dialogues"` to load contexts with both utilities, or `"self_play"` to sample utilities uniformly.

### Seeding
Use Gym's `reset(seed=...)` or call `env.seed(seed)` to control dataset sampling and turn order.

### Action masking
`info["action_mask"]` marks valid act types (e.g., `agree` only valid if partner proposed/insisted in the last turn). `info["oA_max"]` provides per-item upper bounds for `oA`.

### Alignment with literature
- Main source: 2303.00001v1 (DealOrNoDeal negotiation with coarse dialogue acts; training via SL then RL)
- References: Lewis et al., 2017; Kwon et al., 2021

## Configuration

You can pass `NegotiationConfig` via Gym kwargs when registering or making the env:

```python
from deal_or_no_deal_env.env import NegotiationConfig
cfg = NegotiationConfig(use_dataset=True, dataset_script_path="./deal_or_no_dialog_main/deal_or_no_dialog.py")
env_id = register_deal_or_no_deal(kwargs={"config": cfg})
```

## Examples

- Load negotiation dataset: `python -m examples.load_negotiation_dataset`
- Run a random negotiation agent: `python -m examples.random_negotiation_agent`

## References
- Main: 2303.00001v1.pdf
- Reference: D17-1259.pdf


