"""Interactive 3D demo. Run: python main.py"""
from ursina import Ursina, Entity, Text, color, camera, held_keys, time, mouse, window, Vec3, invoke
from ursina.prefabs.editor_camera import EditorCamera
from navigation import Grid, MAP, START
from ai_agent import QLearningAgent

app = Ursina()
window.title = 'Intelligent 3D Avatar Navigation | Q-learning'
window.color = color.rgb(185, 205, 225)
grid = Grid()
agent = QLearningAgent(grid)

# Coordinates are grid cells; character movement is interpolated in world space.
def world(cell, y=0.5):
    return Vec3(cell[0]-grid.width/2, y, cell[1]-grid.height/2)

floor = Entity(model='cube', scale=(grid.width, .1, grid.height), position=world((grid.width/2-.5, grid.height/2-.5), -.1), color=color.rgb(90, 135, 102), collider='box')
for z, row in enumerate(MAP):
    for x, tile in enumerate(row):
        if tile == '#':
            Entity(model='cube', position=world((x,z), .6), scale=(1, 1.2, 1), color=color.rgb(68, 88, 106), collider='box')
        else:
            Entity(model='cube', position=world((x,z), -.02), scale=(.93, .02, .93), color=color.rgb(125, 162, 135) if (x+z)%2 else color.rgb(133, 172, 143))

avatar = Entity(model='cube', position=world(START), scale=(.55,.9,.55), color=color.azure, collider='box')
head = Entity(parent=avatar, model='sphere', y=.65, scale=.9, color=color.rgb(240, 202, 160))
marker = Entity(model='sphere', scale=.35, color=color.orange, enabled=False)
label = Text(text='WASD / arrows: walk  |  Shift: run  |  Click a floor cell: AI navigate  |  R: reset', position=(-.85,.45), scale=.85, color=color.black)
status = Text(text='Mode: idle', position=(-.85,.39), scale=1, color=color.black)

camera.position = (0, 21, -17)
camera.rotation_x = 52
camera.fov = 60
cell = START
route = []
mode = 'idle'
move_cooldown = 0


def set_status(note=''):
    status.text = f'Mode: {mode}  {note}'


def move_to(target, duration):
    global cell
    if not grid.walkable(target) or (abs(cell[0]-target[0])+abs(cell[1]-target[1]) != 1):
        return False
    cell = target
    avatar.animate_position(world(cell), duration=duration, curve='linear')
    avatar.animate_rotation_y(avatar.rotation_y + 90, duration=duration)
    return True


def input(key):
    global cell, mode, route, move_cooldown
    if key == 'r':
        cell = START; route = []; avatar.position = world(cell); mode = 'idle'; marker.enabled = False
        set_status()
    elif key == 'left mouse down' and mouse.hovered_entity and mouse.hovered_entity == floor:
        # Floor raycast intersects large plane, map world position to nearest grid cell.
        if mouse.world_point:
            target = (round(mouse.world_point.x + grid.width/2), round(mouse.world_point.z + grid.height/2))
            select_destination(target)
    elif key == 'left mouse down' and mouse.world_point:
        target = (round(mouse.world_point.x + grid.width/2), round(mouse.world_point.z + grid.height/2))
        if grid.walkable(target):
            select_destination(target)


def select_destination(target):
    global route, mode, move_cooldown
    if not grid.walkable(target):
        set_status('Blocked destination')
        return
    marker.enabled = True
    marker.position = world(target, .25)
    mode = 'training'
    set_status('Training Q-learning policy...')
    agent.train(target)
    planned = agent.learned_path(cell)
    if planned:
        route = planned[1:]
        mode = 'AI (Q-learning)'
    else:
        # Fail-safe A* is explicitly marked, never presented as learned behavior.
        route = grid.shortest_path(cell, target)[1:]
        mode = 'fallback (A*)' if route else 'idle'
    move_cooldown = 0
    set_status(f'{len(route)} steps | trained {agent.episodes} episodes')


def update():
    global mode, route, move_cooldown
    move_cooldown -= time.dt
    keys = {'w':(0,1), 'up arrow':(0,1), 's':(0,-1), 'down arrow':(0,-1),
            'a':(-1,0), 'left arrow':(-1,0), 'd':(1,0), 'right arrow':(1,0)}
    direction = next((v for k,v in keys.items() if held_keys[k]), None)
    running = bool(held_keys['shift'])
    step_time = .12 if running else .23
    if direction:
        if mode not in ('manual walk','manual run'):
            route = []
        mode = 'manual run' if running else 'manual walk'
        if move_cooldown <= 0:
            target = (cell[0]+direction[0],cell[1]+direction[1])
            move_to(target, step_time)
            move_cooldown = step_time
        set_status()
    elif route and move_cooldown <= 0:
        nxt = route.pop(0)
        if move_to(nxt, .18):
            move_cooldown = .19
        if not route:
            set_status('Destination reached')
    elif mode.startswith('manual'):
        mode='idle'; set_status()
    # Simple visible animation state, not skeletal animation.
    avatar.color = color.rgb(255, 175, 75) if mode == 'manual run' else (color.azure if mode != 'idle' else color.rgb(66, 149, 180))

app.run()
