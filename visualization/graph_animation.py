from __future__ import annotations

import pickle
from pathlib import Path

import pygame
import osmnx as ox
from matplotlib import cm
from matplotlib import colormaps
import networkx as nx
import os
import cv2
import numpy as np

DEFAULT_TRAINED_TRAJ_NAME = "trained_trajectory.pkl"
DEFAULT_RANDOM_TRAJ_NAME = "random_trajectory.pkl"


SCREEN_SIZE = (1200, 900)
BG_COLOR = (15, 16, 18)

ROAD_COLORS = {
    "motorway": (255, 90, 90),
    "trunk": (255, 160, 0),
    "primary": (255, 210, 50),
    "secondary": (180, 255, 50),
    "tertiary": (80, 220, 255),
    "residential": (130, 130, 140),
    "service": (80, 80, 100),
    "unclassified": (60, 60, 60),
    "default": (70, 70, 75),
}

#----------------------------------------
# 支持中文字体
def load_label_font(size):
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑 Bold
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
    ]

    for path in candidates:
        if os.path.exists(path):
            return pygame.font.Font(path, size)

    # 兜底：如果都没找到，再退回系统字体
    return pygame.font.SysFont("microsoftyahei", size)

#----------------------------------------
def load_trajectory(path: str | Path):
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def filter_episode(records, episode_idx: int):
    return [r for r in records if r["episode"] == episode_idx]


def load_graph(graph_path: str | Path):
    return ox.load_graphml(graph_path)







def get_screen_coords(x, y, bounds, screen_size):
    min_x, max_x, min_y, max_y = bounds
    w, h = screen_size
    padding = 50
    px = padding + (x - min_x) / (max_x - min_x) * (w - 2 * padding)
    py = h - (padding + (y - min_y) / (max_y - min_y) * (h - 2 * padding))
    return int(px), int(py)


def compute_bounds(graph):
    xs = [data["x"] for _, data in graph.nodes(data=True)]
    ys = [data["y"] for _, data in graph.nodes(data=True)]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max_x - min_x
    span_y = max_y - min_y
    max_span = max(span_x, span_y)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    return (cx - max_span / 2, cx + max_span / 2, cy - max_span / 2, cy + max_span / 2)


def point_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5



def build_map_surface(graph, bounds, screen_size):
    surface = pygame.Surface(screen_size)
    surface.fill(BG_COLOR)

    road_label_count = 0
    max_road_labels = 50
    drawn_names = set()   # 新增：记录已经画过的路名

    font = load_label_font(12)  # 不要循环里重复创建字体

    for u, v, data in graph.edges(data=True):
        hw = data.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]
        color = ROAD_COLORS.get(hw, ROAD_COLORS["default"])
        width = 2 if hw in ["primary", "motorway", "trunk"] else 1

        if "geometry" in data and data["geometry"] is not None:
            points = [get_screen_coords(x, y, bounds, screen_size) for x, y in data["geometry"].coords]
            if len(points) >= 2:
                pygame.draw.lines(surface, color, False, points, width)

                # 画少量道路名 + 去重
                if road_label_count < max_road_labels:
                    road_name = data.get("name") or data.get("ref")
                    if isinstance(road_name, list):
                        road_name = road_name[0]

                    if road_name:
                        road_name = str(road_name).strip()

                        if road_name not in drawn_names:   # 新增：只画一次
                            mid = points[len(points) // 2]
                            lbl = font.render(road_name[:18], True, (150, 150, 150))
                            surface.blit(lbl, (mid[0] + 3, mid[1] + 2))

                            drawn_names.add(road_name)     # 新增：标记已画
                            road_label_count += 1
        else:
            u_node = graph.nodes[u]
            v_node = graph.nodes[v]
            p1 = get_screen_coords(u_node["x"], u_node["y"], bounds, screen_size)
            p2 = get_screen_coords(v_node["x"], v_node["y"], bounds, screen_size)
            pygame.draw.line(surface, color, p1, p2, width)

    return surface


def build_node_screen_positions(graph, bounds, screen_size):
    node_ids = list(graph.nodes)
    node_ids.sort()

    pos = {}
    for node_id in node_ids:
        data = graph.nodes[node_id]
        pos[node_id] = get_screen_coords(data["x"], data["y"], bounds, screen_size)

    return node_ids, pos


def zone_to_screen(zone_idx: int, node_ids, node_screen_pos):
    if zone_idx < 0 or zone_idx >= len(node_ids):
        return None
    node_id = node_ids[zone_idx]
    return node_screen_pos[node_id]


#-------------mapping to full graph-------------------------------
# 训练时只用42个节点，但是可视化我们希望它能尽量沿着真实道路去走

def build_coarse_to_full_mapping(coarse_graph, full_graph):
    """
    Map each coarse graph node to its nearest node in the full graph.
    Returns:
        coarse_node_ids: sorted list of coarse graph node ids
        coarse_to_full: dict[coarse_node_id] -> full_graph_node_id
    """
    coarse_node_ids = list(coarse_graph.nodes)
    coarse_node_ids.sort()

    full_node_ids = list(full_graph.nodes)
    full_node_ids.sort()

    full_xy = {}
    for nid in full_node_ids:
        data = full_graph.nodes[nid]
        full_xy[nid] = (float(data["x"]), float(data["y"]))

    coarse_to_full = {}

    for coarse_id in coarse_node_ids:
        cx = float(coarse_graph.nodes[coarse_id]["x"])
        cy = float(coarse_graph.nodes[coarse_id]["y"])

        best_full_id = None
        best_dist2 = float("inf")

        for full_id in full_node_ids:
            fx, fy = full_xy[full_id]
            dx = fx - cx
            dy = fy - cy
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_full_id = full_id

        coarse_to_full[coarse_id] = best_full_id

    return coarse_node_ids, coarse_to_full


def zone_to_full_node(zone_idx: int, coarse_node_ids, coarse_to_full):
    if zone_idx < 0 or zone_idx >= len(coarse_node_ids):
        return None
    coarse_node_id = coarse_node_ids[zone_idx]
    return coarse_to_full.get(coarse_node_id)



def build_route_screen_points(
    src_zone_idx: int,
    dst_zone_idx: int,
    full_graph,
    coarse_node_ids,
    coarse_to_full,
    full_node_screen_pos,
):
    src_full = zone_to_full_node(src_zone_idx, coarse_node_ids, coarse_to_full)
    dst_full = zone_to_full_node(dst_zone_idx, coarse_node_ids, coarse_to_full)

    if src_full is None or dst_full is None:
        return []

    if src_full == dst_full:
        pos = full_node_screen_pos.get(src_full)
        return [pos] if pos is not None else []

    try:
        path_nodes = nx.shortest_path(full_graph, src_full, dst_full, weight="length")
    except Exception:
        return []

    points = []
    for nid in path_nodes:
        pos = full_node_screen_pos.get(nid)
        if pos is not None:
            if not points or points[-1] != pos:
                points.append(pos)

    return points






#--------------drawing tools-------------------------------------

def draw_fading_tail(screen, path_points, color, 
                     tail_length=40, line_width=6,
                     min_alpha=0.05, max_alpha=0.85,
                     curve="quadratic"):
    if len(path_points) < 2:
        return

    seg_points = path_points[-(tail_length + 1):]
    n = len(seg_points) - 1
    if n <= 0:
        return

    # 预创建一张 surface 复用，避免每段都新建（性能优化）
    temp = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)

    for i in range(n):
        p1 = seg_points[i]
        p2 = seg_points[i + 1]

        t = (i + 1) / n
        if curve == "quadratic":
            alpha = min_alpha + (max_alpha - min_alpha) * (t ** 2)
        elif curve == "sqrt":
            alpha = min_alpha + (max_alpha - min_alpha) * (t ** 0.5)
        else:  # linear
            alpha = min_alpha + (max_alpha - min_alpha) * t

        rgba = (color[0], color[1], color[2], int(255 * alpha))
        pygame.draw.line(temp, rgba, p1, p2, line_width)

    screen.blit(temp, (0, 0))


def draw_pending_hotspots(screen, pending_counts, coarse_node_ids, coarse_to_full, full_node_screen_pos, cmap_name="YlOrRd"):
    if not pending_counts:
        return

    max_pending = max(pending_counts)
    if max_pending <= 0:
        return

    cmap = colormaps.get_cmap(cmap_name)

    for zone_idx, count in enumerate(pending_counts):
        if count <= 0 or zone_idx >= len(coarse_node_ids):
            continue

        coarse_node_id = coarse_node_ids[zone_idx]
        full_node_id = coarse_to_full.get(coarse_node_id)
        if full_node_id is None:
            continue

        pos = full_node_screen_pos.get(full_node_id)
        if pos is None:
            continue

        norm = count / max_pending
        rgba = cmap(norm)
        color = (
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255),
            110,
        )

        temp = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
        radius = 6 + int(12 * norm)
        pygame.draw.circle(temp, color, pos, radius)
        screen.blit(temp, (0, 0))


def draw_legend(screen, font, small_font, car_img):
    panel = pygame.Surface((260, 140), pygame.SRCALPHA)
    panel.fill((20, 20, 24, 180))
    screen.blit(panel, (20, 100))

    title = font.render("Legend", True, (240, 240, 240))
    screen.blit(title, (35, 110))

    pygame.draw.line(screen, (37, 99, 235), (40, 145), (80, 145), 4)
    screen.blit(small_font.render("Taxi trail", True, (220, 220, 220)), (90, 136))

    hotspot = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
    pygame.draw.circle(hotspot, (255, 120, 80, 140), (58, 175), 10)
    screen.blit(hotspot, (0, 0))
    screen.blit(small_font.render("Order hotspot", True, (220, 220, 220)), (90, 166))

    screen.blit(car_img, car_img.get_rect(center=(58, 205)))
    screen.blit(small_font.render("Current taxi", True, (220, 220, 220)), (90, 196))

    pygame.draw.line(screen, (80, 220, 255), (40, 235), (80, 235), 2)
    screen.blit(small_font.render("Road network", True, (220, 220, 220)), (90, 226))


#------------play control-------------------------
def init_playback_state():
    return {
        "paused": False,
        "play_speed": 1,
        "running": True,
    }

def init_interp_state():
    """记录当前正在插值的路段信息"""
    return {
        "seg_points": [],   # 当前 step 的路径点列表
        "sub_idx": 0,       # 当前走到第几个路径点
        "step_idx": 0,      # 对应的 frame_idx（step）
    }


def rebuild_route_points(
    episode_records,
    frame_idx,
    full_graph,
    coarse_node_ids,
    coarse_to_full,
    full_node_screen_pos,
):
    route_points = []

    max_i = min(frame_idx + 1, len(episode_records))
    for i in range(max_i):
        rec = episode_records[i]
        src_zone = int(rec["state"][0])
        dst_zone = int(rec["next_state"][0])

        seg_points = build_route_screen_points(
            src_zone,
            dst_zone,
            full_graph,
            coarse_node_ids,
            coarse_to_full,
            full_node_screen_pos,
        )

        for p in seg_points:
            if p is not None:
                if not route_points or route_points[-1] != p:
                    route_points.append(p)

    return route_points


def handle_playback_events(
    playback,
    frame_idx,
    episode_records,
    full_graph,
    coarse_node_ids,
    coarse_to_full,
    full_node_screen_pos,
):
    route_points = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            playback["running"] = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                playback["paused"] = not playback["paused"]

            elif event.key == pygame.K_RIGHT:
                playback["paused"] = True
                frame_idx = min(frame_idx + 1, len(episode_records) - 1)
                route_points = rebuild_route_points(
                    episode_records,
                    frame_idx,
                    full_graph,
                    coarse_node_ids,
                    coarse_to_full,
                    full_node_screen_pos,
                )

            elif event.key == pygame.K_LEFT:
                playback["paused"] = True
                frame_idx = max(frame_idx - 1, 0)
                route_points = rebuild_route_points(
                    episode_records,
                    frame_idx,
                    full_graph,
                    coarse_node_ids,
                    coarse_to_full,
                    full_node_screen_pos,
                )

            elif event.key == pygame.K_r:
                playback["paused"] = True
                frame_idx = 0
                route_points = rebuild_route_points(
                    episode_records,
                    frame_idx,
                    full_graph,
                    coarse_node_ids,
                    coarse_to_full,
                    full_node_screen_pos,
                )

            elif event.key == pygame.K_UP:
                playback["play_speed"] = min(playback["play_speed"] + 1, 8)

            elif event.key == pygame.K_DOWN:
                playback["play_speed"] = max(playback["play_speed"] - 1, 1)

    return frame_idx, route_points


def advance_frame(playback, frame_idx, episode_len):
    if (not playback["paused"]) and frame_idx < episode_len - 1:
        frame_idx = min(frame_idx + playback["play_speed"], episode_len - 1)
    return frame_idx


def draw_playback_controls(screen, font, playback, screen_size):
    control_text = font.render(
        f"[Space] Pause/Resume  [Left/Right] Step  [Up/Down] Speed={playback['play_speed']}  [R] Reset",
        True,
        (200, 200, 200),
    )
    screen.blit(control_text, (30, screen_size[1] - 35))


















# -----------Core Function----------------------------
def run_graph_animation(
    outputs_dir: str | Path,
    episode_idx: int = 0,
    mode: str = "trained",
    trained_traj_name: str = DEFAULT_TRAINED_TRAJ_NAME,
    random_traj_name: str = DEFAULT_RANDOM_TRAJ_NAME,
    graph_path: str | Path = "cache/taxi_graph_full.graphml",
    coarse_graph_path: str | Path = "cache/taxi_graph.graphml",
    fps: int = 6,
    record: bool = False,
):
    outputs_dir = Path(outputs_dir)
    graph_path = Path(graph_path)
    coarse_graph_path = Path(coarse_graph_path)

    trained_records = load_trajectory(outputs_dir / trained_traj_name)
    random_records = load_trajectory(outputs_dir / random_traj_name)

    trained_episode = filter_episode(trained_records, episode_idx)
    random_episode = filter_episode(random_records, episode_idx)

    if mode == "trained":
        episode_records = trained_episode
        taxi_color = (37, 99, 235)
        title_text = "Trained Agent"
    else:
        episode_records = random_episode
        taxi_color = (249, 115, 22)
        title_text = "Random Agent"

    if not episode_records:
        raise ValueError(f"No records found for episode {episode_idx} in mode={mode}")

    full_graph = load_graph(graph_path)
    coarse_graph = load_graph(coarse_graph_path)

    pygame.init()
    pygame.key.set_repeat(120, 60)

    video_writer = None
    if record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(outputs_dir / "replay.mp4"), fourcc, fps, SCREEN_SIZE)
    
    bounds = compute_bounds(full_graph)
    base_map = build_map_surface(full_graph, bounds, SCREEN_SIZE)

    _, full_node_screen_pos = build_node_screen_positions(full_graph, bounds, SCREEN_SIZE)
    coarse_node_ids, coarse_to_full = build_coarse_to_full_mapping(coarse_graph, full_graph)

    

    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Graph Taxi Replay")
    clock = pygame.time.Clock()
    font = load_label_font(24)
    small_font = load_label_font(20)

    car_img = pygame.image.load("visualization/taxi.png").convert_alpha()
    car_img = pygame.transform.smoothscale(car_img, (28, 28))

    path_points = []

    interp = init_interp_state()
    interp_speed = 0.6   # 每帧沿路段走多少个路径点，调大变快，调小更丝滑

    frame_idx = 0
    playback = init_playback_state()

    while playback["running"]:
        frame_idx, rebuilt_path = handle_playback_events(
            playback,
            frame_idx,
            episode_records,
            full_graph,
            coarse_node_ids,
            coarse_to_full,
            full_node_screen_pos,
        )

        if rebuilt_path is not None:
            path_points = rebuilt_path

        screen.blit(base_map, (0, 0))

        rec_idx = min(frame_idx, len(episode_records) - 1)
        rec = episode_records[rec_idx]

        pending_counts = rec.get("pending_counts", [])
        draw_pending_hotspots(
            screen,
            pending_counts,
            coarse_node_ids,
            coarse_to_full,
            full_node_screen_pos,
            cmap_name="YlOrRd",
        )

        # curr_zone = int(rec["next_state"][0])
        # curr_full_node = zone_to_full_node(curr_zone, coarse_node_ids, coarse_to_full)
        # taxi_pos = full_node_screen_pos.get(curr_full_node) if curr_full_node is not None else None

        # if rec_idx > 0:
        #     prev_rec = episode_records[rec_idx - 1]
        #     prev_zone = int(prev_rec["next_state"][0])
        # else:
        #     prev_zone = int(rec["state"][0])

        # seg_points = build_route_screen_points(
        #     prev_zone,
        #     curr_zone,
        #     full_graph,
        #     coarse_node_ids,
        #     coarse_to_full,
        #     full_node_screen_pos,
        # )

        # for p in seg_points:
        #     if p is not None:
        #         if not path_points or path_points[-1] != p:
        #             path_points.append(p)

        curr_zone = int(rec["next_state"][0])

        if rec_idx > 0:
            prev_rec = episode_records[rec_idx - 1]
            prev_zone = int(prev_rec["next_state"][0])
        else:
            prev_zone = int(rec["state"][0])

        # 检测是否进入新的 step，重新计算路段插值序列
        if interp["step_idx"] != rec_idx:
            seg = build_route_screen_points(
                prev_zone, curr_zone,
                full_graph, coarse_node_ids, coarse_to_full, full_node_screen_pos,
            )
            interp["seg_points"] = seg
            interp["sub_idx"] = 0
            interp["step_idx"] = rec_idx
            interp["prev_sub_idx"] = 0

        seg = interp["seg_points"]

        # 沿路段逐点推进（非暂停时每帧走 interp_speed 个点）
        if not playback["paused"] and len(seg) > 1:
            interp["sub_idx"] = min(
                interp["sub_idx"] + interp_speed,
                len(seg) - 1,
            )

        # 计算小车当前插值位置（在两个路径点之间做线性插值）
        sub_i = interp["sub_idx"]
        if len(seg) == 0:
            taxi_pos = None
        elif sub_i >= len(seg) - 1:
            taxi_pos = seg[-1]
        else:
            # 小数部分做亚像素线性插值
            i_floor = int(sub_i)
            t = sub_i - i_floor
            p1 = seg[i_floor]
            p2 = seg[i_floor + 1]
            taxi_pos = (
                int(p1[0] + t * (p2[0] - p1[0])),
                int(p1[1] + t * (p2[1] - p1[1])),
            )

        # 把已走过的部分追加进 tail
        # for p in seg[: int(interp["sub_idx"]) + 1]:
        #     if p is not None and (not path_points or path_points[-1] != p):
        #         path_points.append(p)

        # 只追加"上一帧sub_idx"到"当前sub_idx"之间新走过的点
        prev_sub = interp.get("prev_sub_idx", 0)
        curr_sub = int(interp["sub_idx"])

        for k in range(prev_sub, curr_sub + 1):
            if k < len(seg):
                p = seg[k]
                if p is not None and (not path_points or path_points[-1] != p):
                    path_points.append(p)

        interp["prev_sub_idx"] = curr_sub


        draw_fading_tail(screen, path_points, taxi_color, tail_length=30)

        if taxi_pos is not None:
            car_rect = car_img.get_rect(center=taxi_pos)
            screen.blit(car_img, car_rect)

        reward = float(rec.get("reward", 0.0))
        matched = bool(rec.get("matched", False))
        total_pending = int(sum(pending_counts)) if pending_counts else 0
        orders_here = int(pending_counts[curr_zone]) if pending_counts and curr_zone < len(pending_counts) else 0

        text1 = font.render(
            f"{title_text} | episode={episode_idx} | step={rec_idx}",
            True,
            (240, 240, 240),
        )
        text2 = small_font.render(
            f"reward={reward:.2f} | matched={matched} | zone={curr_zone} | orders_here={orders_here} | total_pending={total_pending}",
            True,
            (220, 220, 220),
        )

        screen.blit(text1, (30, 25))
        screen.blit(text2, (30, 60))

        draw_legend(screen, font, small_font, car_img)
        draw_playback_controls(screen, small_font, playback, SCREEN_SIZE)

        pygame.display.flip()
        clock.tick(fps)

        if record:
            arr = pygame.surfarray.array3d(screen)
            frame = cv2.cvtColor(np.transpose(arr, (1, 0, 2)), cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        # frame_idx = advance_frame(
        #     playback,
        #     frame_idx,
        #     len(episode_records),
        # )

        # 只有插值到路段末尾才自动推进下一帧
        seg = interp["seg_points"]
        if not playback["paused"]:
            if len(seg) == 0 or interp["sub_idx"] >= len(seg) - 1:
                frame_idx = advance_frame(playback, frame_idx, len(episode_records))

    pygame.quit()

    if record and video_writer is not None:
        video_writer.release()


if __name__ == "__main__":
    run_graph_animation(
        outputs_dir="outputs/run_20260420_113532",
        episode_idx=99,
        mode="trained",
        graph_path="cache/taxi_graph_full.graphml",
        coarse_graph_path="cache/taxi_graph.graphml",
        fps=6,
        record=True
    )