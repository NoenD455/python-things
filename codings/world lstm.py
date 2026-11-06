import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import threading
import queue
import random
import math
import heapq


# ---------- Maze generation ----------
def generate_maze(w=15, h=15, seed=None):
    if seed is not None:
        random.seed(seed)
    maze = np.ones((h, w), dtype=np.uint8)
    stack = [(1, 1)]
    maze[1, 1] = 0
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx < w - 1 and 1 <= ny < h - 1 and maze[ny, nx] == 1:
                neighbors.append((nx, ny, dx, dy))
        if neighbors:
            nx, ny, dx, dy = random.choice(neighbors)
            maze[y + dy // 2, x + dx // 2] = 0
            maze[ny, nx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()
    return maze


def try_move(maze, px, py, dx, dy):
    h, w = maze.shape
    new_x = px + dx
    if 0 <= new_x < w and maze[int(py), int(new_x)] == 0:
        px = new_x
    new_y = py + dy
    if 0 <= new_y < h and maze[int(new_y), int(px)] == 0:
        py = new_y
    return px, py


def astar_path(maze, start, goal):
    h, w = maze.shape
    if start == goal:
        return [start]
    if maze[goal[1], goal[0]] != 0:
        return None
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0:
                new_cost = cost_so_far[current] + 1
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + abs(nx - goal[0]) + abs(ny - goal[1])
                    heapq.heappush(frontier, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current
    if goal not in came_from:
        return None
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path


# ---------- Raycaster ----------
class Raycaster:
    def __init__(self, maze, resolution=32, fov=math.pi / 3, supersample=2):
        self.maze = maze
        self.res = resolution
        self.fov = fov
        self.supersample = supersample
        self.h, self.w = maze.shape

    def cast(self, px, py, theta):
        ss = self.supersample
        res = self.res * ss
        half = res // 2

        img = np.zeros((res, res, 3), dtype=np.float32)
        img[:half, :] = np.array([50, 50, 70], dtype=np.float32)
        img[half:, :] = np.array([30, 30, 40], dtype=np.float32)

        dir_x = math.cos(theta)
        dir_y = math.sin(theta)
        plane_len = math.tan(self.fov / 2)
        plane_x = -dir_y * plane_len
        plane_y = dir_x * plane_len

        for x in range(res):
            camera_x = 2 * x / res - 1
            ray_dir_x = dir_x + plane_x * camera_x
            ray_dir_y = dir_y + plane_y * camera_x

            map_x = int(px)
            map_y = int(py)
            delta_dist_x = abs(1 / ray_dir_x) if ray_dir_x != 0 else 1e30
            delta_dist_y = abs(1 / ray_dir_y) if ray_dir_y != 0 else 1e30

            if ray_dir_x < 0:
                step_x = -1
                side_dist_x = (px - map_x) * delta_dist_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1 - px) * delta_dist_x
            if ray_dir_y < 0:
                step_y = -1
                side_dist_y = (py - map_y) * delta_dist_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1 - py) * delta_dist_y

            side = 0
            perp_dist = 1e30
            hit_map_x, hit_map_y = map_x, map_y
            for _ in range(4 * (self.w + self.h)):
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side = 0
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side = 1
                if map_x < 0 or map_x >= self.w or map_y < 0 or map_y >= self.h:
                    perp_dist = 1e30
                    break
                elif self.maze[map_y, map_x] == 1:
                    if side == 0:
                        perp_dist = side_dist_x - delta_dist_x
                    else:
                        perp_dist = side_dist_y - delta_dist_y
                    hit_map_x, hit_map_y = map_x, map_y
                    break

            perp_dist = max(perp_dist, 0.01)
            line_height = int(res / perp_dist)
            draw_start = max(0, half - line_height // 2)
            draw_end = min(res - 1, half + line_height // 2)

            checker = (hit_map_x + hit_map_y) & 1
            if side == 0:
                base = (np.array([205, 165, 120], dtype=np.float32) if checker
                        else np.array([145, 105, 70], dtype=np.float32))
            else:
                base = (np.array([175, 185, 215], dtype=np.float32) if checker
                        else np.array([115, 125, 160], dtype=np.float32))
            shade = max(0.25, 1.0 - perp_dist / 18.0)
            img[draw_start:draw_end + 1, x] = base * shade

        if ss > 1:
            img = img.reshape(self.res, ss, self.res, ss, 3).mean(axis=(1, 3))
        return img.clip(0, 255).astype(np.uint8)


# ---------- Quantization ----------
def quantize(z, z_min, z_max, num_bins):
    range_ = (z_max - z_min).clamp(min=1e-6)
    normalized = (z - z_min) / range_
    return (normalized * num_bins).floor().long().clamp(0, num_bins - 1)


def dequantize(bins, z_min, z_max, num_bins):
    return z_min + (bins.float() + 0.5) / num_bins * (z_max - z_min)


def decode_logits(logits, z_min, z_max, num_bins):
    return dequantize(logits.argmax(dim=-1), z_min, z_max, num_bins)


# ---------- VAE ----------
class VAE(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        c = base_channels
        self.c4 = c * 4
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.ReLU(),
        )
        self.enc_out_size = c * 4 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_out_size, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_size)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(c, in_channels, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.reshape(h.size(0), self.c4, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ---------- Base game ----------
class BaseGame:
    name = "Base"

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        """Returns (frame_uint8_HWC, reward_float)."""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def new_level(self):
        return self.reset()

    def auto_action(self):
        return np.random.uniform(-1, 1, size=2).astype(np.float32)

    def status_text(self):
        return ""


# ---------- Maze base (shared A* auto-gen) ----------
class MazeBase(BaseGame):
    def __init__(self, seed=42):
        self.maze = generate_maze(15, 15, seed=seed)
        self.speed = 0.08
        self.rot_speed = 0.25

    def reset(self):
        self.px, self.py, self.theta = 1.5, 1.5, 0.0
        self._auto_target_cell = (1, 1)
        self._auto_path = []
        self._auto_last_cell = None
        self._auto_aim_x = self.px
        self._auto_aim_y = self.py
        self._auto_wiggle = 0
        self._auto_stuck_counter = 0
        return self.render()

    def step(self, action):
        dx = self.speed * math.cos(self.theta) * action[0]
        dy = self.speed * math.sin(self.theta) * action[0]
        self.px, self.py = try_move(self.maze, self.px, self.py, dx, dy)
        self.theta = (self.theta + self.rot_speed * action[1]) % (2 * math.pi)
        return self.render(), 0.0

    def _on_maze_rebuilt(self):
        pass

    def new_level(self):
        self.maze = generate_maze(15, 15, seed=random.randint(0, 99999))
        self._on_maze_rebuilt()
        return self.reset()

    def _pick_random_cell(self, min_dist=0, from_cell=None):
        h, w = self.maze.shape
        for _ in range(200):
            x = random.randint(1, w - 2)
            y = random.randint(1, h - 2)
            if self.maze[y, x] != 0:
                continue
            if from_cell is not None:
                d = abs(x - from_cell[0]) + abs(y - from_cell[1])
                if d < min_dist:
                    continue
            return (x, y)
        return (1, 1)

    def auto_action(self):
        cx, cy = int(self.px), int(self.py)

        if (cx, cy) != self._auto_last_cell:
            self._auto_last_cell = (cx, cy)

            need_replan = (not self._auto_path
                           or (cx, cy) == self._auto_target_cell
                           or self._auto_path[0] != (cx, cy))
            if need_replan:
                if (cx, cy) == self._auto_target_cell or not self._auto_path:
                    for _ in range(50):
                        t = self._pick_random_cell(min_dist=4, from_cell=(cx, cy))
                        if astar_path(self.maze, (cx, cy), t) is not None:
                            self._auto_target_cell = t
                            break
                new_path = astar_path(self.maze, (cx, cy), self._auto_target_cell)
                self._auto_path = new_path if new_path else [(cx, cy)]

            if len(self._auto_path) >= 2:
                nxt = self._auto_path[1]
                aim_x = nxt[0] + 0.5
                aim_y = nxt[1] + 0.5
                dx = aim_x - self.px
                dy = aim_y - self.py
                dist = math.hypot(dx, dy)
                if dist > 1e-3:
                    max_off = math.tan(math.radians(15)) * dist
                    off = random.uniform(-max_off, max_off)
                    perp_x = -dy / dist
                    perp_y = dx / dist
                    aim_x += perp_x * off
                    aim_y += perp_y * off
                self._auto_aim_x = aim_x
                self._auto_aim_y = aim_y
            else:
                self._auto_aim_x = self.px + 0.5 * math.cos(self.theta)
                self._auto_aim_y = self.py + 0.5 * math.sin(self.theta)

            if random.random() < 0.3:
                self._auto_wiggle = random.choice([-1, 1]) * random.randint(1, 3)

        desired = math.atan2(self._auto_aim_y - self.py,
                             self._auto_aim_x - self.px)
        error = (desired - self.theta + math.pi) % (2 * math.pi) - math.pi

        action = np.zeros(2, dtype=np.float32)
        if self._auto_wiggle != 0:
            action[1] = 1.0 if self._auto_wiggle > 0 else -1.0
            self._auto_wiggle -= 1
            action[0] = 1.0
        elif abs(error) > 0.5:
            action[1] = 1.0 if error > 0 else -1.0
            action[0] = 0.0
        else:
            action[1] = max(-1.0, min(1.0, error * 2.0))
            action[0] = 1.0

        return action

    def status_text(self):
        return f"Pos: ({self.px:.2f}, {self.py:.2f})"


# ---------- Maze 2D ----------
class Maze2D(MazeBase):
    name = "2D Maze"

    def __init__(self, seed=42):
        super().__init__(seed=seed)
        self.reset()

    def render(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        cell = 32 / self.maze.shape[0]
        for y in range(self.maze.shape[0]):
            for x in range(self.maze.shape[1]):
                y0 = int(y * cell)
                y1 = min(32, int((y + 1) * cell) + 1)
                x0 = int(x * cell)
                x1 = min(32, int((x + 1) * cell) + 1)
                if self.maze[y, x] == 1:
                    img[y0:y1, x0:x1] = [220, 200, 160]
                else:
                    img[y0:y1, x0:x1] = [30, 30, 40]
        px_s = int(self.px * cell)
        py_s = int(self.py * cell)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                yy, xx = py_s + dy, px_s + dx
                if 0 <= yy < 32 and 0 <= xx < 32:
                    img[yy, xx] = [255, 50, 50]
        fx = int((self.px + 0.5 * math.cos(self.theta)) * cell)
        fy = int((self.py + 0.5 * math.sin(self.theta)) * cell)
        if 0 <= fy < 32 and 0 <= fx < 32:
            img[fy, fx] = [255, 240, 0]
        return img


# ---------- Maze 3D ----------
class Maze3D(MazeBase):
    name = "3D Maze"

    def __init__(self, seed=42, raycaster_res=32, supersample=2):
        super().__init__(seed=seed)
        self.raycaster_res = raycaster_res
        self.supersample = supersample
        self.raycaster = Raycaster(self.maze, resolution=raycaster_res, supersample=supersample)
        self.reset()

    def _on_maze_rebuilt(self):
        self.raycaster = Raycaster(self.maze, resolution=self.raycaster_res,
                                   supersample=self.supersample)

    def render(self):
        return self.raycaster.cast(self.px, self.py, self.theta)


# ---------- DVD Dodge ----------
class DVDDodge(BaseGame):
    name = "DVD Dodge"

    def __init__(self):
        self.reset()

    def reset(self):
        self.player_x = 16.0
        self.player_y = 26.0
        while True:
            self.dvd_x = random.uniform(4, 28)
            self.dvd_y = random.uniform(4, 28)
            if abs(self.dvd_x - self.player_x) > 8 or abs(self.dvd_y - self.player_y) > 8:
                break
        self.dvd_vx = random.choice([-0.8, 0.8])
        self.dvd_vy = random.choice([-0.8, 0.8])
        return self.render()

    def step(self, action):
        self.player_y -= action[0] * 0.9
        self.player_x += action[1] * 0.9
        self.player_x = float(np.clip(self.player_x, 2, 29))
        self.player_y = float(np.clip(self.player_y, 2, 29))

        self.dvd_x += self.dvd_vx
        self.dvd_y += self.dvd_vy
        if self.dvd_x <= 2 or self.dvd_x >= 28:
            self.dvd_vx = -self.dvd_vx
            self.dvd_x = float(np.clip(self.dvd_x, 2, 28))
        if self.dvd_y <= 2 or self.dvd_y >= 28:
            self.dvd_vy = -self.dvd_vy
            self.dvd_y = float(np.clip(self.dvd_y, 2, 28))
        return self.render(), 0.0

    def render(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:] = [20, 20, 40]
        x0 = int(self.dvd_x) - 2
        y0 = int(self.dvd_y) - 2
        img[max(0, y0):min(32, y0 + 4), max(0, x0):min(32, x0 + 4)] = [80, 120, 255]
        px0 = int(self.player_x) - 1
        py0 = int(self.player_y) - 1
        img[max(0, py0):min(32, py0 + 3), max(0, px0):min(32, px0 + 3)] = [255, 220, 50]
        return img

    def status_text(self):
        return (f"Player: ({self.player_x:.1f},{self.player_y:.1f})  "
                f"DVD: ({self.dvd_x:.1f},{self.dvd_y:.1f})")


# ---------- Square Jump (auto-bounce doodle jump) ----------
class SquareJump(BaseGame):
    name = "Square Jump"

    def __init__(self):
        self.reset()

    def reset(self):
        self.player_x = 16.0
        self.player_y = 22.0
        self.vy = 0.0
        self.platforms = [(12, 26, 8), (4, 20, 6), (22, 16, 6), (10, 10, 6), (0, 4, 6)]
        self.score = 0
        return self.render()

    def step(self, action):
        self.player_x = (self.player_x + action[1] * 1.2) % 32

        self.vy += 0.2
        self.player_y += self.vy

        if self.vy > 0:
            for px, py, pw in self.platforms:
                if px <= self.player_x <= px + pw:
                    if py - 0.5 <= self.player_y <= py + 1.5:
                        self.player_y = py
                        self.vy = -3.0
                        break

        if self.player_y < 16:
            shift = 16 - self.player_y
            self.player_y = 16
            self.platforms = [(px, py + shift, pw) for px, py, pw in self.platforms]

        self.platforms = [p for p in self.platforms if p[1] < 36]
        while len(self.platforms) < 6:
            top_y = min(p[1] for p in self.platforms) if self.platforms else 32
            self.platforms.append((random.randint(0, 26), top_y - 5, 6))

        if self.player_y > 34:
            self.reset()

        self.score = max(self.score, int(-self.player_y))
        return self.render(), 0.0

    def render(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:] = [40, 50, 80]
        for px, py, pw in self.platforms:
            y0 = int(py)
            x0 = int(px)
            if 0 <= y0 < 32:
                img[y0, max(0, x0):min(32, x0 + pw)] = [50, 200, 80]
        px0 = int(self.player_x) - 1
        py0 = int(self.player_y) - 1
        img[max(0, py0):min(32, py0 + 3), max(0, px0):min(32, px0 + 3)] = [220, 50, 50]
        return img

    def status_text(self):
        return f"Score: {self.score}"


# ---------- Light World ----------
class LightWorld(BaseGame):
    name = "Light World"

    def __init__(self):
        self.speed = 0.12
        self.rot_speed = 0.22
        self.reset()

    def reset(self):
        self.px = 16.0
        self.py = 16.0
        self.theta = 0.0
        self.obstacles = [
            (6, 6, 4, 4),
            (22, 6, 4, 4),
            (6, 22, 4, 4),
            (22, 22, 4, 4),
            (14, 4, 4, 2),
        ]
        return self.render()

    def step(self, action):
        new_px = self.px + self.speed * math.cos(self.theta) * action[0]
        new_py = self.py + self.speed * math.sin(self.theta) * action[0]
        for ox, oy, ow, oh in self.obstacles:
            if ox - 0.5 <= new_px <= ox + ow + 0.5 and oy - 0.5 <= new_py <= oy + oh + 0.5:
                if not (ox - 0.5 <= new_px <= ox + ow + 0.5
                        and oy - 0.5 <= self.py <= oy + oh + 0.5):
                    new_py = self.py
                else:
                    new_px = self.px
                break
        self.px = float(np.clip(new_px, 1, 31))
        self.py = float(np.clip(new_py, 1, 31))
        self.theta = (self.theta + self.rot_speed * action[1]) % (2 * math.pi)
        return self.render(), 0.0

    def render(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        obstacle_mask = np.zeros((32, 32), dtype=bool)
        for ox, oy, ow, oh in self.obstacles:
            obstacle_mask[oy:oy + oh, ox:ox + ow] = True

        fov = math.radians(70)
        num_rays = 80
        for i in range(num_rays):
            angle = self.theta - fov / 2 + fov * i / num_rays
            dx = math.cos(angle)
            dy = math.sin(angle)
            for d in np.arange(0.5, 18.0, 1.0):
                x = self.px + d * dx
                y = self.py + d * dy
                if not (0 <= x < 32 and 0 <= y < 32):
                    break
                xi, yi = int(x), int(y)
                if obstacle_mask[yi, xi]:
                    break
                intensity = max(0.1, 1.0 - d / 14.0)
                new_r = int(255 * intensity)
                if img[yi, xi, 0] < new_r:
                    img[yi, xi] = [new_r, new_r, int(180 * intensity)]

        img[obstacle_mask] = [70, 50, 40]
        img[int(self.py), int(self.px)] = [255, 80, 80]
        return img

    def status_text(self):
        return f"Pos: ({self.px:.1f},{self.py:.1f})  θ: {math.degrees(self.theta):.0f}°"


GAMES = {
    "2D Maze": lambda: Maze2D(seed=random.randint(0, 99999)),
    "3D Maze": lambda: Maze3D(seed=42),
    "DVD Dodge": DVDDodge,
    "Square Jump": SquareJump,
    "Light World": LightWorld,
}


# ---------- Recurrent World Model ----------
class RecurrentWorldModel(nn.Module):
    def __init__(self, latent_dim, action_dim, rnn_type='lstm', hidden_dim=128,
                 num_layers=2, num_bins=64, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.rnn_type = rnn_type.lower()
        assert self.rnn_type in ('lstm', 'gru', 'rnn')
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.z_min = None
        self.z_max = None

        self.input_proj = nn.Linear(latent_dim + action_dim, hidden_dim)
        rnn_cls = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}[self.rnn_type]
        actual_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(hidden_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=actual_dropout)
        self.readout = nn.Linear(hidden_dim, latent_dim * num_bins)

    def init_hidden(self, batch_size, device):
        shape = (self.num_layers, batch_size, self.hidden_dim)
        if self.rnn_type == 'lstm':
            return (torch.zeros(*shape, device=device),
                    torch.zeros(*shape, device=device))
        return torch.zeros(*shape, device=device)

    def forward(self, z, a, hidden=None):
        single_step = z.dim() == 2
        if single_step:
            z = z.unsqueeze(1)
            a = a.unsqueeze(1)

        B = z.shape[0]
        if hidden is None:
            hidden = self.init_hidden(B, z.device)

        x = self.input_proj(torch.cat([z, a], dim=-1))
        out, hidden = self.rnn(x, hidden)
        logits = self.readout(out).view(B, z.shape[1], self.latent_dim, self.num_bins)

        if single_step:
            logits = logits.squeeze(1)
        return logits, hidden


def detach_hidden(hidden):
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(h.detach() for h in hidden)
    return hidden.detach()


# ---------- Rollout dataset ----------
class TransitionDatasetRollout(torch.utils.data.Dataset):
    def __init__(self, z_t, actions, z_next, unroll=10,
                 z_t_flip=None, z_next_flip=None):
        self.z_t = z_t
        self.actions = actions
        self.z_next = z_next
        self.unroll = unroll
        self.z_t_flip = z_t_flip
        self.z_next_flip = z_next_flip
        self.n = max(0, len(z_t) - unroll + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.z_t_flip is not None and random.random() < 0.5:
            a_seq = self.actions[i:i + self.unroll].clone()
            a_seq[:, 1] = -a_seq[:, 1]
            return (self.z_t_flip[i:i + self.unroll],
                    a_seq,
                    self.z_next_flip[i:i + self.unroll])
        return (self.z_t[i:i + self.unroll],
                self.actions[i:i + self.unroll],
                self.z_next[i:i + self.unroll])


# ---------- App ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Dream Maze - VAE + Recurrent World Model")
        root.geometry("1000x900")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.game = GAMES["3D Maze"]()
        self.frames = []
        self.actions = []
        self.keys = {}
        self.recording = False
        self.last_frame = None
        self.vae = None
        self.wm = None
        self.z = None
        self.wm_hidden = None
        self.inference_running = False
        self.advance_one_frame = False
        self.vae_stop_event = None
        self.wm_stop_event = None
        self.vae_queue = None
        self.wm_queue = None

        self.auto_gen_running = False
        self.auto_gen_step = 0
        self.auto_gen_target = 0

        self.record_photo = None
        self.inference_photo = None
        self.vae_photo = None
        self.wm_photo = None

        self.setup_ui()
        root.bind('<KeyPress>', self.on_key_press)
        root.bind('<KeyRelease>', self.on_key_release)
        self.render_record_view()
        self._update_game_info()

    # ========== UI ==========
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.record_tab = ttk.Frame(notebook)
        notebook.add(self.record_tab, text="Record")
        self.setup_record_tab()

        self.vae_tab = ttk.Frame(notebook)
        notebook.add(self.vae_tab, text="Train VAE")
        self.setup_vae_tab()

        self.wm_tab = ttk.Frame(notebook)
        notebook.add(self.wm_tab, text="Train World Model")
        self.setup_wm_tab()

        self.settings_tab = ttk.Frame(notebook)
        notebook.add(self.settings_tab, text="Settings")
        self.setup_settings_tab()

        self.inference_tab = ttk.Frame(notebook)
        notebook.add(self.inference_tab, text="Inference")
        self.setup_inference_tab()

    def setup_record_tab(self):
        frame = ttk.Frame(self.record_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.record_canvas = tk.Canvas(frame, width=512, height=512, bg='black')
        self.record_canvas.pack(side=tk.LEFT, padx=10)

        panel = ttk.Frame(frame)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(panel, text="Game:").pack(anchor=tk.W)
        self.game_name_var = tk.StringVar(value=self.game.name)
        self.game_combo = ttk.Combobox(panel, textvariable=self.game_name_var,
                                       values=list(GAMES.keys()), state='readonly')
        self.game_combo.pack(fill=tk.X, pady=(0, 8))
        self.game_combo.bind('<<ComboboxSelected>>', self.on_game_change)

        self.record_status = ttk.Label(panel, text="Frames: 0")
        self.record_status.pack(anchor=tk.W, pady=2)
        self.game_info_label = ttk.Label(panel, text="", foreground="gray")
        self.game_info_label.pack(anchor=tk.W, pady=(0, 6))

        ttk.Button(panel, text="Start Recording", command=self.start_recording).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Stop Recording", command=self.stop_recording).pack(fill=tk.X, pady=2)
        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Button(panel, text="Auto-Generate Dataset", command=self.auto_generate).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Stop Auto-Generate", command=self.stop_auto_generate).pack(fill=tk.X, pady=2)
        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Button(panel, text="Clear Data", command=self.clear_data).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="New Level", command=self.new_level).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Save Data", command=self.save_data).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Load Data", command=self.load_data).pack(fill=tk.X, pady=2)

        self.record_canvas.bind('<Button-1>', lambda e: self.record_canvas.focus_set())

    def setup_vae_tab(self):
        frame = ttk.Frame(self.vae_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.vae_canvas = tk.Canvas(frame, width=512, height=256, bg='black')
        self.vae_canvas.pack(pady=10)
        self.vae_progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.vae_progress.pack(pady=5)
        self.vae_status = ttk.Label(frame, text="Not trained")
        self.vae_status.pack(pady=5)
        btn = ttk.Frame(frame)
        btn.pack(pady=5)
        ttk.Button(btn, text="Start Training", command=self.start_vae_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn, text="Stop Training", command=self.stop_vae_training).pack(side=tk.LEFT, padx=5)

    def setup_wm_tab(self):
        frame = ttk.Frame(self.wm_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.wm_canvas = tk.Canvas(frame, width=512, height=256, bg='black')
        self.wm_canvas.pack(pady=10)
        self.wm_progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.wm_progress.pack(pady=5)
        self.wm_status = ttk.Label(frame, text="Not trained")
        self.wm_status.pack(pady=5)
        self.wm_config_label = ttk.Label(frame, text="", foreground="gray")
        self.wm_config_label.pack(pady=2)
        btn = ttk.Frame(frame)
        btn.pack(pady=5)
        ttk.Button(btn, text="Start Training", command=self.start_wm_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn, text="Stop Training", command=self.stop_wm_training).pack(side=tk.LEFT, padx=5)

    def setup_settings_tab(self):
        frame = ttk.Frame(self.settings_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        vae_frame = ttk.LabelFrame(frame, text="VAE Settings")
        vae_frame.pack(fill=tk.X, pady=5)
        ttk.Label(vae_frame, text="Base channels:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.vae_base_var = tk.IntVar(value=16)
        ttk.Combobox(vae_frame, textvariable=self.vae_base_var,
                     values=[8, 16, 32, 64]).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(vae_frame, text="Latent dim:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.vae_latent_var = tk.IntVar(value=32)
        ttk.Combobox(vae_frame, textvariable=self.vae_latent_var,
                     values=[8, 16, 32, 64, 128]).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(vae_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.vae_epochs_var = tk.IntVar(value=50)
        ttk.Entry(vae_frame, textvariable=self.vae_epochs_var).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(vae_frame, text="Learning rate:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.vae_lr_var = tk.DoubleVar(value=1e-3)
        ttk.Entry(vae_frame, textvariable=self.vae_lr_var).grid(row=3, column=1, padx=5, pady=2)

        wm_frame = ttk.LabelFrame(frame, text="Recurrent World Model")
        wm_frame.pack(fill=tk.X, pady=5)
        ttk.Label(wm_frame, text="RNN type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.rnn_type_var = tk.StringVar(value="LSTM")
        ttk.Combobox(wm_frame, textvariable=self.rnn_type_var,
                     values=["LSTM", "GRU", "RNN"], state='readonly').grid(
            row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(wm_frame, text="Hidden dim:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_hidden_var = tk.IntVar(value=128)
        ttk.Combobox(wm_frame, textvariable=self.wm_hidden_var,
                     values=[32, 64, 128, 256]).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(wm_frame, text="Num layers:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_layers_var = tk.IntVar(value=2)
        ttk.Combobox(wm_frame, textvariable=self.wm_layers_var,
                     values=[1, 2, 3]).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(wm_frame, text="Dropout (layers>1):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_dropout_var = tk.DoubleVar(value=0.0)
        ttk.Combobox(wm_frame, textvariable=self.wm_dropout_var,
                     values=[0.0, 0.1, 0.2, 0.3]).grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(wm_frame, text="Unroll length (T):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_unroll_var = tk.IntVar(value=10)
        ttk.Combobox(wm_frame, textvariable=self.wm_unroll_var,
                     values=[2, 5, 10, 20, 40]).grid(row=4, column=1, padx=5, pady=2)
        ttk.Label(wm_frame, text="Epochs:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_epochs_var = tk.IntVar(value=50)
        ttk.Entry(wm_frame, textvariable=self.wm_epochs_var).grid(row=5, column=1, padx=5, pady=2)
        ttk.Label(wm_frame, text="Learning rate:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        self.wm_lr_var = tk.DoubleVar(value=1e-3)
        ttk.Entry(wm_frame, textvariable=self.wm_lr_var).grid(row=6, column=1, padx=5, pady=2)

        aug_frame = ttk.LabelFrame(frame, text="Augmentation & Regularization")
        aug_frame.pack(fill=tk.X, pady=5)
        ttk.Label(aug_frame, text="Mirror augmentation:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.mirror_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(aug_frame, variable=self.mirror_var).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(aug_frame, text="Stillness loss weight:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.stillness_var = tk.DoubleVar(value=0.1)
        ttk.Entry(aug_frame, textvariable=self.stillness_var).grid(row=1, column=1, padx=5, pady=2)

        lat_frame = ttk.LabelFrame(frame, text="Latent Representation")
        lat_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lat_frame, text="Bins per dim:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.num_bins_var = tk.IntVar(value=64)
        ttk.Combobox(lat_frame, textvariable=self.num_bins_var,
                     values=[8, 16, 32, 64, 128]).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(lat_frame, text="(Quantized cross-entropy on all timesteps)",
                  foreground="gray").grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

    def setup_inference_tab(self):
        frame = ttk.Frame(self.inference_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.inference_canvas = tk.Canvas(frame, width=512, height=512, bg='black')
        self.inference_canvas.pack(side=tk.LEFT, padx=10)
        panel = ttk.Frame(frame)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.inference_status = ttk.Label(panel, text="Ready")
        self.inference_status.pack(anchor=tk.W, pady=5)

        # --- Frame wait controls ---
        wait_frame = ttk.LabelFrame(panel, text="Frame Control")
        wait_frame.pack(fill=tk.X, pady=5)
        self.frame_wait_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(wait_frame, text="Frame wait (step per input)",
                        variable=self.frame_wait_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Button(wait_frame, text="Step (0 action)",
                   command=self.step_one_frame).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(panel, text="Start Dream", command=self.start_inference).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Stop Dream", command=self.stop_inference).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Reset Dream", command=self.reset_inference).pack(fill=tk.X, pady=2)
        ttk.Label(panel, text=("Arrow keys control the dream.\n"
                               "Frame wait pauses between steps;\n"
                               "each key press (or hold) advances\n"
                               "one frame.\n\n"
                               "Reset Dream re-encodes the\n"
                               "current view and clears hidden."),
                  justify=tk.LEFT).pack(anchor=tk.W, pady=10)
        self.inference_canvas.bind('<Button-1>', lambda e: self.inference_canvas.focus_set())
        self.display_frame(self.game.render(), self.inference_canvas, 'inference_photo')

    def _update_game_info(self):
        name = self.game.name
        if name in ("2D Maze", "3D Maze"):
            info = "Up/Down: forward/back\nLeft/Right: turn"
        elif name == "DVD Dodge":
            info = "Arrows: move in 2D\nAvoid the blue square."
        elif name == "Square Jump":
            info = "Left/Right: move\nAuto-bounce on platforms."
        elif name == "Light World":
            info = "Up/Down: forward/back\nLeft/Right: turn"
        else:
            info = ""
        self.game_info_label.config(text=info)

    def display_frame(self, frame, canvas, photo_attr, size=512):
        img = Image.fromarray(frame).resize((size, size), Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        setattr(self, photo_attr, photo)
        canvas.delete("frame")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo, tags="frame")

    def render_record_view(self):
        self.display_frame(self.game.render(), self.record_canvas, 'record_photo')

    def on_key_press(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry)):
            return
        if event.keysym in ['Up', 'Down', 'Left', 'Right']:
            self.keys[event.keysym] = True
            self.advance_one_frame = True
            return "break"

    def on_key_release(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry)):
            return
        if event.keysym in ['Up', 'Down', 'Left', 'Right']:
            self.keys[event.keysym] = False
            return "break"

    def read_keys(self):
        a = np.zeros(2, dtype=np.float32)
        if self.keys.get('Up'): a[0] = 1.0
        if self.keys.get('Down'): a[0] = -1.0
        if self.keys.get('Left'): a[1] = -1.0
        if self.keys.get('Right'): a[1] = 1.0
        return a

    # ========== Game switching ==========
    def on_game_change(self, event=None):
        if self.recording or self.auto_gen_running or self.inference_running:
            self.game_name_var.set(self.game.name)
            messagebox.showwarning("Warning",
                "Stop recording / auto-gen / dreaming before switching games.")
            return
        new_name = self.game_name_var.get()
        if new_name == self.game.name:
            return
        if self.frames and not messagebox.askyesno(
                "Switch game",
                "Switching games clears recorded data and trained models. Continue?"):
            self.game_name_var.set(self.game.name)
            return
        self.frames = []
        self.actions = []
        self.last_frame = None
        self.vae = None
        self.wm = None
        self.z = None
        self.wm_hidden = None
        self.game = GAMES[new_name]()
        self.game.reset()
        self.render_record_view()
        self._update_game_info()
        self.record_status.config(text=f"Frames: 0  Game: {new_name}")
        self.vae_status.config(text="Not trained")
        self.wm_status.config(text="Not trained")
        self.wm_config_label.config(text="")
        self.wm_progress['value'] = 0
        self.vae_progress['value'] = 0

    def new_level(self):
        if self.recording:
            self.stop_recording()
        if self.auto_gen_running:
            self.stop_auto_generate()
        if self.inference_running:
            self.stop_inference()
        self.game.new_level()
        self.frames = []
        self.actions = []
        self.last_frame = None
        self.z = None
        self.wm_hidden = None
        self.render_record_view()
        self.record_status.config(text=f"Frames: 0  New level ({self.game.name})")

    # ========== Record ==========
    def start_recording(self):
        if self.auto_gen_running:
            self.stop_auto_generate()
        self.frames = []
        self.actions = []
        self.last_frame = None
        self.recording = True
        self.record_tick()

    def stop_recording(self):
        self.recording = False

    def record_tick(self):
        if not self.recording:
            return
        frame = self.game.render()
        if self.last_frame is None:
            self.frames.append(frame)
        action = self.read_keys()
        self.game.step(action)
        new_frame = self.game.render()
        self.frames.append(new_frame)
        self.actions.append(action)
        self.last_frame = new_frame
        self.display_frame(new_frame, self.record_canvas, 'record_photo')
        status = self.game.status_text()
        suffix = f"  {status}" if status else ""
        self.record_status.config(text=f"Frames: {len(self.frames)}{suffix}")
        self.root.after(40, self.record_tick)

    # ========== Auto-generate ==========
    def auto_generate(self):
        if self.auto_gen_running:
            return
        if self.recording:
            self.stop_recording()

        n_steps = simpledialog.askinteger(
            "Auto-Generate Dataset",
            "Number of steps to generate:",
            initialvalue=5000, minvalue=100, maxvalue=200000,
            parent=self.root)
        if n_steps is None:
            return

        self.frames = []
        self.actions = []
        self.game.reset()

        self.auto_gen_running = True
        self.auto_gen_step = 0
        self.auto_gen_target = n_steps

        frame = self.game.render()
        self.frames.append(frame)
        self.last_frame = frame
        self.auto_gen_tick()

    def stop_auto_generate(self):
        self.auto_gen_running = False
        self.record_status.config(text=f"Auto-gen stopped. Frames: {len(self.frames)}")

    def auto_gen_tick(self):
        for _ in range(5):
            if not self.auto_gen_running or self.auto_gen_step >= self.auto_gen_target:
                self.auto_gen_running = False
                self.record_status.config(text=f"Auto-gen done. Frames: {len(self.frames)}")
                if self.last_frame is not None:
                    self.display_frame(self.last_frame, self.record_canvas, 'record_photo')
                return
            self.auto_gen_step_once()

        if self.last_frame is not None:
            self.display_frame(self.last_frame, self.record_canvas, 'record_photo')
        self.record_status.config(
            text=f"Auto-gen: {self.auto_gen_step}/{self.auto_gen_target}  "
                 f"Frames: {len(self.frames)}")
        self.root.after(1, self.auto_gen_tick)

    def auto_gen_step_once(self):
        old_px = getattr(self.game, 'px', None)
        old_py = getattr(self.game, 'py', None)
        action = self.game.auto_action()
        new_frame, _ = self.game.step(action)

        if old_px is not None and hasattr(self.game, '_auto_stuck_counter') and action[0] > 0:
            moved = (abs(self.game.px - old_px) > 1e-4
                     or abs(self.game.py - old_py) > 1e-4)
            if not moved:
                self.game._auto_stuck_counter += 1
            else:
                self.game._auto_stuck_counter = 0
            if self.game._auto_stuck_counter > 10:
                self.game._auto_last_cell = None
                self.game._auto_stuck_counter = 0
                self.game.theta = random.uniform(0, 2 * math.pi)

        self.frames.append(new_frame)
        self.actions.append(action)
        self.last_frame = new_frame
        self.auto_gen_step += 1

    # ========== Data I/O ==========
    def clear_data(self):
        if self.auto_gen_running:
            self.stop_auto_generate()
        self.frames = []
        self.actions = []
        self.last_frame = None
        self.record_status.config(text="Frames: 0")

    def save_data(self):
        if len(self.frames) == 0:
            messagebox.showinfo("Info", "No data to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NPZ", "*.npz")])
        if path:
            np.savez_compressed(path, frames=np.stack(self.frames), actions=np.stack(self.actions))
            messagebox.showinfo("Info", f"Saved {len(self.frames)} frames to {path}")

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("NPZ", "*.npz")])
        if path:
            data = np.load(path)
            self.frames = list(data['frames'])
            self.actions = list(data['actions'])
            self.last_frame = self.frames[-1] if self.frames else None
            self.recording = False
            self.record_status.config(text=f"Loaded {len(self.frames)} frames")

    # ========== Preview helper ==========
    def _make_compare_grid(self, left_t, right_t, size=128):
        left = left_t.detach().cpu().permute(0, 2, 3, 1).numpy()
        right = right_t.detach().cpu().permute(0, 2, 3, 1).numpy()
        n = left.shape[0]
        grid = np.zeros((n * size, 2 * size, 3), dtype=np.uint8)
        for i in range(n):
            l = (left[i] * 255).clip(0, 255).astype(np.uint8)
            r = (right[i] * 255).clip(0, 255).astype(np.uint8)
            grid[i * size:(i + 1) * size, :size] = np.array(
                Image.fromarray(l).resize((size, size), Image.NEAREST))
            grid[i * size:(i + 1) * size, size:] = np.array(
                Image.fromarray(r).resize((size, size), Image.NEAREST))
        return Image.fromarray(grid)

    def _preview_schedule(self, epochs):
        freq = 5 if epochs >= 5 else 1
        def should(epoch):
            return epoch == 0 or (epoch + 1) % freq == 0 or epoch == epochs - 1
        return should

    # ========== VAE training ==========
    def start_vae_training(self):
        if len(self.frames) < 20:
            messagebox.showerror("Error", "Need at least 20 frames. Record or auto-generate first.")
            return
        if self.vae_stop_event is not None:
            self.vae_stop_event.set()

        base_ch = self.vae_base_var.get()
        latent_dim = self.vae_latent_var.get()
        epochs = self.vae_epochs_var.get()
        lr = self.vae_lr_var.get()
        mirror = self.mirror_var.get()

        frames = np.stack(self.frames).astype(np.float32) / 255.0
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(frames_t), batch_size=32, shuffle=True)

        preview_rng = np.random.RandomState(42)
        preview_idx = preview_rng.choice(len(frames), min(4, len(frames)), replace=False)
        preview_frames_cpu = frames_t[preview_idx]

        self.vae = VAE(in_channels=3, base_channels=base_ch, latent_dim=latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)
        stop_event = threading.Event()
        self.vae_stop_event = stop_event
        q = queue.Queue()
        self.vae_queue = q
        vae = self.vae
        should_preview = self._preview_schedule(epochs)

        def train():
            for epoch in range(epochs):
                if stop_event.is_set():
                    q.put(('done', True))
                    return
                total_loss = 0
                for batch in loader:
                    x = batch[0].to(self.device)
                    if mirror and random.random() < 0.5:
                        x = torch.flip(x, dims=[3])
                    recon, mu, logvar = vae(x)
                    recon_loss = F.mse_loss(recon, x)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.001 * kl_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(loader)
                q.put(('progress', epoch + 1, epochs, avg_loss))

                if should_preview(epoch):
                    pv = preview_frames_cpu.to(self.device)
                    with torch.no_grad():
                        recon_pv, _, _ = vae(pv)
                    img = self._make_compare_grid(pv, recon_pv)
                    q.put(('preview', img))
            q.put(('done', False))

        threading.Thread(target=train, daemon=True).start()
        self.poll_vae_queue()

    def stop_vae_training(self):
        if self.vae_stop_event is not None:
            self.vae_stop_event.set()

    def poll_vae_queue(self):
        q = self.vae_queue
        if q is None:
            return
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == 'progress':
                    _, epoch, total, loss = msg
                    self.vae_progress['value'] = epoch / total * 100
                    self.vae_status.config(text=f"Epoch {epoch}/{total}  Loss: {loss:.4f}")
                elif msg[0] == 'preview':
                    img = msg[1]
                    photo = ImageTk.PhotoImage(img)
                    self.vae_photo = photo
                    self.vae_canvas.delete("all")
                    self.vae_canvas.config(width=img.width, height=img.height)
                    self.vae_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                elif msg[0] == 'done':
                    if msg[1]:
                        self.vae_status.config(text="Stopped")
                    else:
                        self.vae_progress['value'] = 100
                        self.vae_status.config(text="Training complete!")
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.poll_vae_queue)

    # ========== World model training ==========
    def start_wm_training(self):
        if self.vae is None:
            messagebox.showerror("Error", "Train VAE first.")
            return
        if len(self.actions) < 20:
            messagebox.showerror("Error", "Need at least 20 transitions.")
            return
        if self.wm_stop_event is not None:
            self.wm_stop_event.set()

        rnn_type = self.rnn_type_var.get().lower()
        hidden_dim = self.wm_hidden_var.get()
        num_layers = self.wm_layers_var.get()
        dropout = float(self.wm_dropout_var.get())
        unroll = int(self.wm_unroll_var.get())
        epochs = self.wm_epochs_var.get()
        lr = self.wm_lr_var.get()
        mirror = self.mirror_var.get()
        stillness_w = float(self.stillness_var.get())
        num_bins = int(self.num_bins_var.get())
        assert num_bins > 0

        if len(self.actions) < unroll:
            messagebox.showerror(
                "Error",
                f"Need at least {unroll} transitions for unroll length {unroll}; "
                f"have {len(self.actions)}.")
            return

        self.vae.eval()
        frames = np.stack(self.frames).astype(np.float32) / 255.0
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

        latents = []
        with torch.no_grad():
            for i in range(0, len(frames_t), 64):
                mu, _ = self.vae.encode(frames_t[i:i + 64].to(self.device))
                latents.append(mu.cpu())
        latents = torch.cat(latents)

        if mirror:
            frames_flip = torch.flip(frames_t, dims=[3])
            latents_flip = []
            with torch.no_grad():
                for i in range(0, len(frames_flip), 64):
                    mu, _ = self.vae.encode(frames_flip[i:i + 64].to(self.device))
                    latents_flip.append(mu.cpu())
            latents_flip = torch.cat(latents_flip)
        else:
            latents_flip = None

        actions_t = torch.from_numpy(np.stack(self.actions).astype(np.float32))
        z_t = latents[:-1]
        z_next = latents[1:]
        z_t_flip = latents_flip[:-1] if latents_flip is not None else None
        z_next_flip = latents_flip[1:] if latents_flip is not None else None

        dataset = TransitionDatasetRollout(
            z_t, actions_t, z_next, unroll=unroll,
            z_t_flip=z_t_flip, z_next_flip=z_next_flip)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        self.wm = RecurrentWorldModel(
            latent_dim=self.vae.latent_dim,
            action_dim=2,
            rnn_type=rnn_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_bins=num_bins,
            dropout=dropout,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.wm.parameters() if p.requires_grad)
        self.wm_config_label.config(
            text=(f"rnn={rnn_type.upper()}  hidden={hidden_dim}  layers={num_layers}  "
                  f"dropout={dropout:.2f}  unroll={unroll}  bins={num_bins}  "
                  f"params={num_params:,}  latents={self.vae.latent_dim}"))

        all_lat = torch.cat([latents, latents_flip], 0) if latents_flip is not None else latents
        z_min = all_lat.min(dim=0).values
        z_max = all_lat.max(dim=0).values
        margin = (z_max - z_min) * 0.05 + 1e-4
        self.wm.z_min = (z_min - margin).to(self.device)
        self.wm.z_max = (z_max + margin).to(self.device)

        optimizer = torch.optim.Adam(self.wm.parameters(), lr=lr)
        stop_event = threading.Event()
        self.wm_stop_event = stop_event
        q = queue.Queue()
        self.wm_queue = q
        wm = self.wm
        vae = self.vae

        preview_rng = np.random.RandomState(42)
        preview_idx = preview_rng.choice(len(actions_t), min(4, len(actions_t)), replace=False)
        preview_idx_list = preview_idx.tolist()
        preview_z_cpu = latents[preview_idx]
        preview_a_cpu = actions_t[preview_idx]
        preview_next_raw_cpu = frames_t[[i + 1 for i in preview_idx_list]]
        should_preview = self._preview_schedule(epochs)

        def train():
            for epoch in range(epochs):
                if stop_event.is_set():
                    q.put(('done', True))
                    return
                total_loss = 0
                for batch in loader:
                    z, a, z_next_seq = [b.to(self.device) for b in batch]

                    input_bins = quantize(z, wm.z_min, wm.z_max, num_bins)
                    z_in = dequantize(input_bins, wm.z_min, wm.z_max, num_bins)
                    target_bins = quantize(z_next_seq, wm.z_min, wm.z_max, num_bins)

                    logits, _ = wm(z_in, a)
                    recon_loss = F.cross_entropy(
                        logits.reshape(-1, num_bins),
                        target_bins.reshape(-1))

                    zero_a = torch.zeros_like(a)
                    still_logits, _ = wm(z_in, zero_a)
                    still_loss = F.cross_entropy(
                        still_logits.reshape(-1, num_bins),
                        input_bins.reshape(-1))

                    loss = recon_loss + stillness_w * still_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                q.put(('progress', epoch + 1, epochs, avg_loss))

                if should_preview(epoch):
                    pz = preview_z_cpu.to(self.device)
                    pa = preview_a_cpu.to(self.device)
                    pnext = preview_next_raw_cpu.to(self.device)
                    with torch.no_grad():
                        pbins = quantize(pz, wm.z_min, wm.z_max, num_bins)
                        pz_in = dequantize(pbins, wm.z_min, wm.z_max, num_bins)
                        logits, _ = wm(pz_in, pa)
                        z_next_pred = decode_logits(logits, wm.z_min, wm.z_max, num_bins)
                        pred_frames = vae.decode(z_next_pred)
                    img = self._make_compare_grid(pnext, pred_frames)
                    q.put(('preview', img))
            q.put(('done', False))

        threading.Thread(target=train, daemon=True).start()
        self.poll_wm_queue()

    def stop_wm_training(self):
        if self.wm_stop_event is not None:
            self.wm_stop_event.set()

    def poll_wm_queue(self):
        q = self.wm_queue
        if q is None:
            return
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == 'progress':
                    _, epoch, total, loss = msg
                    self.wm_progress['value'] = epoch / total * 100
                    self.wm_status.config(text=f"Epoch {epoch}/{total}  Loss: {loss:.4f}")
                elif msg[0] == 'preview':
                    img = msg[1]
                    photo = ImageTk.PhotoImage(img)
                    self.wm_photo = photo
                    self.wm_canvas.delete("all")
                    self.wm_canvas.config(width=img.width, height=img.height)
                    self.wm_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                elif msg[0] == 'done':
                    if msg[1]:
                        self.wm_status.config(text="Stopped")
                    else:
                        self.wm_progress['value'] = 100
                        self.wm_status.config(text="Training complete!")
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.poll_wm_queue)

    # ========== Inference ==========
    def _encode_frame_to_z(self):
        frame = self.game.render()
        frame_t = (torch.from_numpy(frame.astype(np.float32) / 255.0)
                   .permute(2, 0, 1).contiguous().unsqueeze(0).to(self.device))
        with torch.no_grad():
            mu, _ = self.vae.encode(frame_t)
            z = mu
            if self.wm.num_bins > 0:
                bins = quantize(z, self.wm.z_min, self.wm.z_max, self.wm.num_bins)
                z = dequantize(bins, self.wm.z_min, self.wm.z_max, self.wm.num_bins)
            self.z = z

    def start_inference(self):
        if self.vae is None or self.wm is None:
            messagebox.showerror("Error", "Train VAE and world model first.")
            return
        if self.inference_running:
            return
        if self.recording:
            self.stop_recording()
        if self.auto_gen_running:
            self.stop_auto_generate()
        self.vae.eval()
        self.wm.eval()
        self.wm_hidden = None
        self.advance_one_frame = False
        self._encode_frame_to_z()
        self.inference_running = True
        wait = "frame-wait" if self.frame_wait_var.get() else "continuous"
        self.inference_status.config(text=f"Dreaming... ({self.wm.rnn_type.upper()}, {wait})")
        self.inference_tick()

    def stop_inference(self):
        self.inference_running = False
        self.inference_status.config(text="Stopped")

    def reset_inference(self):
        if self.vae is None or self.wm is None:
            return
        self.vae.eval()
        self.wm.eval()
        self.wm_hidden = None
        self._encode_frame_to_z()
        self.inference_status.config(text="Dream reset")

    def step_one_frame(self):
        """Force one frame advance in frame-wait mode with zero action."""
        self.advance_one_frame = True

    def inference_tick(self):
        if not self.inference_running:
            return

        if self.frame_wait_var.get():
            any_key_held = any(self.keys.values())
            if not any_key_held and not self.advance_one_frame:
                self.root.after(30, self.inference_tick)
                return
            self.advance_one_frame = False

        action = self.read_keys()
        a_t = torch.from_numpy(action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.wm.num_bins > 0:
                bins = quantize(self.z, self.wm.z_min, self.wm.z_max, self.wm.num_bins)
                z_in = dequantize(bins, self.wm.z_min, self.wm.z_max, self.wm.num_bins)
            else:
                z_in = self.z

            logits, self.wm_hidden = self.wm(z_in, a_t, self.wm_hidden)
            self.wm_hidden = detach_hidden(self.wm_hidden)
            z_next = decode_logits(logits, self.wm.z_min, self.wm.z_max, self.wm.num_bins)
            recon = self.vae.decode(z_next)

        self.z = z_next
        frame = (recon.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                 ).clip(0, 255).astype(np.uint8)
        self.display_frame(frame, self.inference_canvas, 'inference_photo')
        self.root.after(50, self.inference_tick)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()