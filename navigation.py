"""Grid world, collision checks, and optional A* safety route."""
from __future__ import annotations
from heapq import heappush, heappop

MAP = [
    '###############',
    '#S....#.......#',
    '#.##..#..###..#',
    '#......#......#',
    '#..##.....##..#',
    '#......#......#',
    '#.###..#..##..#',
    '#.............#',
    '#..##..#..##..#',
    '#......#......#',
    '#.##.......##.#',
    '#.....#.......#',
    '###############',
]
# Note all rows must be equal width
assert len({len(line) for line in MAP}) == 1
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
START = next((x, z) for z, row in enumerate(MAP) for x, ch in enumerate(row) if ch == 'S')

class Grid:
    def __init__(self, layout=MAP):
        self.layout = layout
        self.width = len(layout[0]); self.height = len(layout)
        assert all(len(row) == self.width for row in layout)

    def walkable(self, cell):
        x, z = cell
        return 0 <= x < self.width and 0 <= z < self.height and self.layout[z][x] != '#'

    def move(self, cell, action):
        dx, dz = DIRECTIONS[action]
        nxt = (cell[0] + dx, cell[1] + dz)
        return nxt if self.walkable(nxt) else cell

    def neighbors(self, cell):
        for action in range(4):
            nxt = self.move(cell, action)
            if nxt != cell:
                yield action, nxt

    def shortest_path(self, start, goal):
        if not self.walkable(start) or not self.walkable(goal):
            return []
        frontier = [(0, start)]
        cost = {start: 0}; previous = {}
        while frontier:
            _, cur = heappop(frontier)
            if cur == goal:
                path = [cur]
                while cur != start:
                    cur = previous[cur]
                    path.append(cur)
                return list(reversed(path))
            for _, nxt in self.neighbors(cur):
                new_cost = cost[cur] + 1
                if new_cost < cost.get(nxt, float('inf')):
                    cost[nxt] = new_cost; previous[nxt] = cur
                    heuristic = abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                    heappush(frontier, (new_cost + heuristic, nxt))
        return []
