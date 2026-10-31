# Intelligent 3D Avatar Navigation — Reinforcement Learning

**Author:** Yassir EL HAj Tirari

A playable Python 3D maze with manual character movement and an autonomous navigation agent. Unlike a purely scripted A* demo, the agent **trains a tabular Q-learning policy** for each selected destination and attempts to follow it. If the learned route fails, the interface explicitly labels a reliable **A* fallback**.

## Features
- **Manual control:** WASD or arrow keys, and Shift to run.
- **AI navigation:** click a free square to set a destination; the agent trains through reinforcement learning (Q-learning).
- **Obstacle avoidance:** training penalizes moves into walls; autonomous moves stay on traversable cells.
- **Collision prevention:** grid validation blocks illegal moves; walls also have 3D colliders.
- **Movement feedback:** visibly different idle/walking/running states and smooth step transitions. The geometric character does **not** have skeletal walk/run animations.
- **Route recovery:** an explicitly identified A* fallback if the learned Q-policy cannot find a safe route.

## Setup
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python main.py
```

## Controls
| Input | Action |
|---|---|
| W/A/S/D or arrows | Walk |
| Shift + direction | Run |
| Left click on an open cell | Train AI and navigate |
| R | Reset to starting position |

## How the AI works

The agent observes its **grid position**, chooses among four actions (up, down, left, right), and receives rewards: negative for time steps, larger penalties for hitting walls, and a positive terminal reward for reaching the goal. A **Q-learning update** learns action values through 3,500 simulated training episodes per selected target. The avatar then tries the learned route. If it gets stuck, A* finds a deterministic backup route; the on-screen status identifies which algorithm is controlling the avatar. The environment is discrete even though it is rendered in 3D.

## Project files
- `main.py`: playable Ursina scene and input controls
- `navigation.py`: grid map, collision checks, A* fallback
- `ai_agent.py`: actual Q-learning training and learned policy
- `tests/test_navigation.py`: basic verification tests
- `requirements.txt`: dependencies

## Limitations
This is a self-contained **educational prototype**, not an advanced embodied AI. It uses tabular reinforcement learning on a fixed maze, not deep RL, computer vision, physics-based character control, or skeletal animations. Training occurs synchronously and may briefly pause the interface after each click. Change the `MAP` in `navigation.py` to customize obstacles.
