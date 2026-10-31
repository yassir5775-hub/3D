"""Tabular Q-learning agent: a genuinely learned policy for a target cell."""
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, grid, seed=12):
        self.grid = grid
        self.rng = random.Random(seed)
        self.q = defaultdict(lambda: [0.0] * 4)
        self.goal = None
        self.episodes = 0

    def train(self, goal, episodes=3500, max_steps=180):
        """Train a goal-conditioned policy; reward shaping accelerates convergence."""
        if not self.grid.walkable(goal):
            raise ValueError('Destination is blocked.')
        self.goal = goal
        self.q.clear()
        cells = [(x,z) for z in range(self.grid.height) for x in range(self.grid.width)
                 if self.grid.walkable((x,z)) and (x,z) != goal]
        for episode in range(episodes):
            state = self.rng.choice(cells)
            epsilon = max(0.04, 0.9 * (1 - episode / episodes))
            for _ in range(max_steps):
                actions = list(range(4))
                if self.rng.random() < epsilon:
                    action = self.rng.choice(actions)
                else:
                    action = self._best_action(state)
                nxt = self.grid.move(state, action)
                dist_before = abs(goal[0]-state[0]) + abs(goal[1]-state[1])
                dist_after = abs(goal[0]-nxt[0]) + abs(goal[1]-nxt[1])
                reward = -1.0 + 0.5 * (dist_before - dist_after)
                if nxt == state:
                    reward -= 4
                if nxt == goal:
                    reward += 70
                old = self.q[state][action]
                target = reward + (0 if nxt == goal else 0.95 * max(self.q[nxt]))
                self.q[state][action] = old + 0.25 * (target - old)
                state = nxt
                if state == goal:
                    break
        self.episodes = episodes

    def _best_action(self, state):
        scores = self.q[state]
        best = max(scores)
        choices = [a for a, value in enumerate(scores) if value == best]
        return self.rng.choice(choices)

    def learned_path(self, start, max_steps=250):
        """Return a successful learned trajectory, or [] if training did not converge."""
        if self.goal is None:
            return []
        path = [start]
        current = start
        seen = {start}
        for _ in range(max_steps):
            if current == self.goal:
                return path
            # Prefer learned action values; exclude invalid / already visited cells.
            ranked = sorted(range(4), key=lambda a: self.q[current][a], reverse=True)
            choices = [self.grid.move(current, a) for a in ranked]
            nxt = next((c for c in choices if c != current and c not in seen), None)
            if nxt is None:
                return []
            current = nxt
            seen.add(current)
            path.append(current)
        return []
