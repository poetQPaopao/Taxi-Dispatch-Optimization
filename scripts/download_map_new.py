# import pygame
# import osmnx as ox
# import networkx as nx
# import threading
# import sys
# import os

# # --- 配置参数 ---
# # 目标坐标（港科广红鸟广场附近的真实 WGS-84 坐标）
# TARGET_COORDS = (22.894, 113.478)
# # 视界半径，1500米既能覆盖校园周边，又不会导致投影死机
# VIEW_RADIUS = 3000
# # 路口合并容差（米），15米是城市路网抽象的黄金数值
# INTERSECTION_TOLERANCE = 20.0
# SCREEN_SIZE = (1000, 800)
# BG_COLOR = (15, 16, 18)
# ACCENT_COLOR = (0, 200, 255) 
# SHOW_NAMED_NODES = True

# ROAD_COLORS = {
#     'motorway': (255, 90, 90),
#     'trunk': (255, 160, 0),
#     'primary': (255, 210, 50),
#     'secondary': (180, 255, 50),
#     'tertiary': (80, 220, 255),
#     'residential': (130, 130, 140),
#     'service': (80, 80, 100),
#     'unclassified': (60, 60, 60),
#     'default': (70, 70, 75)
# }

# class MapLoader:
#     def __init__(self):
#         self.graph = None              # coarse graph for RL
#         self.full_graph = None         # full graph for visualization
#         self.is_loading = True
#         self.status_text = "Init..."

#     def _get_largest_strongly_connected_component(self, G):
#         largest_scc = max(nx.strongly_connected_components(G), key=len)
#         return G.subgraph(largest_scc).copy()

#     def download_map(self):
#         try:
#             self.status_text = "Downloading HKUST-GZ Map..."
#             ox.settings.use_cache = True

#             os.makedirs("cache", exist_ok=True)

#             # --- 1. 下载较完整的 drive 路网 ---
#             G = ox.graph_from_point(
#                 TARGET_COORDS,
#                 dist=VIEW_RADIUS + 500,
#                 custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"]',
#                 simplify=True,
#             )
#             print(f"[Step 1] 原始下载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

#             # --- 2. 提取连通图 ---
#             G = self._get_largest_strongly_connected_component(G)
#             print(f"[Step 2] 提取连通图后: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

#             # --- 3. 投影 full graph ---
#             self.status_text = f"Projecting full graph..."
#             G_full = ox.project_graph(G)

#             # 再清一次，确保 full graph 干净
#             G_full = self._get_largest_strongly_connected_component(G_full)

#             self.full_graph = G_full
#             ox.save_graphml(G_full, "cache/taxi_graph_full.graphml")
#             print(f"[Step 3] 保存 full graph: {G_full.number_of_nodes()} 节点, {G_full.number_of_edges()} 边")

#             # --- 4. 从 full graph 构建 coarse graph ---
#             self.status_text = "Consolidating Intersections..."
#             G_coarse = ox.consolidate_intersections(
#                 G_full,
#                 tolerance=INTERSECTION_TOLERANCE,
#                 rebuild_graph=True,
#                 dead_ends=False,
#             )
#             print(f"[Step 4] 合并路口后: {G_coarse.number_of_nodes()} 节点, {G_coarse.number_of_edges()} 边")

#             # --- 5. 最终清理 coarse graph ---
#             G_coarse = self._get_largest_strongly_connected_component(G_coarse)
#             print(f"[Step 5] 最终 coarse graph: {G_coarse.number_of_nodes()} 节点, {G_coarse.number_of_edges()} 边")

#             self.graph = G_coarse
#             ox.save_graphml(G_coarse, "cache/taxi_graph.graphml")
#             print("[Done] 已保存:")
#             print("  - cache/taxi_graph_full.graphml")
#             print("  - cache/taxi_graph.graphml")

#             self.is_loading = False
#             self.status_text = "Done!"

#         except Exception as e:
#             self.status_text = f"Error: {e}"
#             print(f"Loading Error: {e}")

# def get_screen_coords(x, y, bounds, screen_size):
#     min_x, max_x, min_y, max_y = bounds
#     w, h = screen_size
#     padding = 50
#     px = padding + (x - min_x) / (max_x - min_x) * (w - 2 * padding)
#     py = h - (padding + (y - min_y) / (max_y - min_y) * (h - 2 * padding))
#     return int(px), int(py)

# def load_label_font(size):
#     candidates = [
#         "/System/Library/Fonts/PingFang.ttc",
#         "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
#         "/System/Library/Fonts/Supplemental/STHeiti Medium.ttc",
#         "/Library/Fonts/Arial Unicode.ttf"
#     ]
#     for path in candidates:
#         if os.path.exists(path):
#             return pygame.font.Font(path, size)
#     for name in ["PingFang SC", "Heiti SC", "Arial Unicode MS"]:
#         path = pygame.font.match_font(name)
#         if path:
#             return pygame.font.Font(path, size)
#     return pygame.font.SysFont(None, size)

# def main():
#     pygame.init()
#     screen = pygame.display.set_mode(SCREEN_SIZE)
#     pygame.display.set_caption("HKUST-GZ RL Environment Map")
#     clock = pygame.time.Clock()
#     font = load_label_font(18)

#     loader = MapLoader()
#     threading.Thread(target=loader.download_map, daemon=True).start()

#     map_surface = None
#     bounds = None
#     angle = 0

#     while True:
#         screen.fill(BG_COLOR)
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit(); sys.exit()

#         if loader.is_loading:
#             # 绘制加载动画
#             angle += 5
#             rect = pygame.Rect(SCREEN_SIZE[0]//2-30, SCREEN_SIZE[1]//2-30, 60, 60)
#             pygame.draw.arc(screen, ACCENT_COLOR, rect, angle*0.017, (angle+120)*0.017, 3)
#             txt = font.render(loader.status_text, True, (150, 150, 150))
#             screen.blit(txt, (SCREEN_SIZE[0]//2 - txt.get_width()//2, SCREEN_SIZE[1]//2 + 50))
#         else:
#             if map_surface is None and loader.graph is not None:
#                 # --- 1. 自动计算完美的屏幕视野 (Bounding Box) ---
#                 xs = [data['x'] for node, data in loader.graph.nodes(data=True)]
#                 ys = [data['y'] for node, data in loader.graph.nodes(data=True)]
                
#                 min_x, max_x = min(xs), max(xs)
#                 min_y, max_y = min(ys), max(ys)
                
#                 span_x = max_x - min_x
#                 span_y = max_y - min_y
#                 max_span = max(span_x, span_y)
                
#                 cx = (min_x + max_x) / 2
#                 cy = (min_y + max_y) / 2
#                 bounds = (cx - max_span/2, cx + max_span/2, cy - max_span/2, cy + max_span/2)

#                 map_surface = pygame.Surface(SCREEN_SIZE)
#                 map_surface.fill(BG_COLOR)
                
#                 # --- 2. 绘制真实的弯曲道路 ---
#                 for u, v, data in loader.graph.edges(data=True):
#                     hw = data.get('highway', 'unclassified')
#                     if isinstance(hw, list): hw = hw[0]
#                     color = ROAD_COLORS.get(hw, ROAD_COLORS['default'])
#                     width = 2 if hw in ['primary', 'motorway', 'trunk'] else 1
                    
#                     # 优先读取 geometry 绘制曲线
#                     if 'geometry' in data:
#                         points = [get_screen_coords(x, y, bounds, SCREEN_SIZE) for x, y in data['geometry'].coords]
#                         if len(points) >= 2:
#                             pygame.draw.lines(map_surface, color, False, points, width)
#                     else:
#                         u_node = loader.graph.nodes[u]
#                         v_node = loader.graph.nodes[v]
#                         p1 = get_screen_coords(u_node['x'], u_node['y'], bounds, SCREEN_SIZE)
#                         p2 = get_screen_coords(v_node['x'], v_node['y'], bounds, SCREEN_SIZE)
#                         pygame.draw.line(map_surface, color, p1, p2, width)

#                 # --- 3. 绘制最终的 RL Agent 枢纽节点 (Hubs) ---
#                 for node_id, data in loader.graph.nodes(data=True):
#                     px, py = get_screen_coords(data['x'], data['y'], bounds, SCREEN_SIZE)
#                     if 0 <= px < SCREEN_SIZE[0] and 0 <= py < SCREEN_SIZE[1]:
#                         # 用明显的白色圆点标注 Hub
#                         pygame.draw.circle(map_surface, (255, 255, 255), (px, py), 5)
                        
#                         # 绘制路名 (如果有)
#                         name = data.get('name') or data.get('ref')
#                         if SHOW_NAMED_NODES and name:
#                             # 处理列表类型的名字
#                             if isinstance(name, list):
#                                 name = name[0]
#                             lbl = font.render(str(name), True, (220, 220, 220))
#                             map_surface.blit(lbl, (px + 8, py + 3))

#             if map_surface:
#                 screen.blit(map_surface, (0, 0))
                
#                 # 绘制图例
#                 for i, (type_name, color) in enumerate(ROAD_COLORS.items()):
#                     pygame.draw.rect(screen, color, (20, 20 + i*20, 15, 10))
#                     lbl = font.render(type_name, True, (200, 200, 200))
#                     screen.blit(lbl, (45, 15 + i*20))

#         pygame.display.flip()
#         clock.tick(60)

# if __name__ == "__main__":
#     main()


import math
import os
import sys
import threading

import networkx as nx
import osmnx as ox
import pygame


# =========================
# 配置参数
# =========================
TARGET_COORDS = (22.894, 113.478)   # HKUST(GZ) 红鸟广场附近
VIEW_RADIUS = 3000                  # 米
INTERSECTION_TOLERANCE = 20.0       # 路口合并容差
SCREEN_SIZE = (1200, 900)
BG_COLOR = (15, 16, 18)
ACCENT_COLOR = (0, 200, 255)

SHOW_NAMED_NODES = True
SHOW_POIS = True
SHOW_ROAD_LABELS = True
SHOW_POI_LABELS = True

MAX_POI_LABELS = 80                 # 防止屏幕太乱
MAX_ROAD_LABELS = 60
MIN_ROAD_LABEL_LENGTH = 120         # 屏幕上道路太短就不标名
POI_MIN_PIXEL_DISTANCE = 28         # POI 标签最小间距


ROAD_COLORS = {
    "motorway": (255, 90, 90),
    "trunk": (255, 160, 0),
    "primary": (255, 210, 50),
    "secondary": (180, 255, 50),
    "tertiary": (80, 220, 255),
    "residential": (130, 130, 140),
    "service": (100, 100, 120),
    "unclassified": (80, 80, 90),
    "default": (70, 70, 75),
}

POI_COLORS = {
    "school": (255, 220, 120),
    "university": (255, 220, 120),
    "restaurant": (255, 170, 120),
    "cafe": (255, 200, 150),
    "bus_station": (120, 255, 180),
    "parking": (180, 180, 255),
    "building": (220, 180, 255),
    "default": (255, 255, 120),
}


# =========================
# 工具函数
# =========================
def safe_first(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def get_screen_coords(x, y, bounds, screen_size):
    min_x, max_x, min_y, max_y = bounds
    w, h = screen_size
    padding = 50

    dx = max(max_x - min_x, 1e-9)
    dy = max(max_y - min_y, 1e-9)

    px = padding + (x - min_x) / dx * (w - 2 * padding)
    py = h - (padding + (y - min_y) / dy * (h - 2 * padding))
    return int(px), int(py)


def load_label_font(size):
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return pygame.font.Font(path, size)

    for name in ["PingFang SC", "Heiti SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)

    return pygame.font.SysFont(None, size)


def point_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def choose_poi_color(row):
    amenity = safe_first(row.get("amenity"))
    building = safe_first(row.get("building"))

    if amenity in POI_COLORS:
        return POI_COLORS[amenity]
    if building:
        return POI_COLORS["building"]
    return POI_COLORS["default"]


def get_edge_highway(data):
    hw = data.get("highway", "default")
    return safe_first(hw) if hw is not None else "default"


def get_edge_name(data):
    name = safe_first(data.get("name"))
    ref = safe_first(data.get("ref"))
    return name or ref


def polyline_midpoint(points):
    if not points:
        return None

    if len(points) == 1:
        return points[0]

    lengths = []
    total = 0.0
    for i in range(len(points) - 1):
        seg_len = point_distance(points[i], points[i + 1])
        lengths.append(seg_len)
        total += seg_len

    if total <= 1e-9:
        return points[len(points) // 2]

    target = total / 2
    accum = 0.0
    for i, seg_len in enumerate(lengths):
        if accum + seg_len >= target:
            ratio = (target - accum) / max(seg_len, 1e-9)
            x = points[i][0] + ratio * (points[i + 1][0] - points[i][0])
            y = points[i][1] + ratio * (points[i + 1][1] - points[i][1])
            return int(x), int(y)
        accum += seg_len

    return points[-1]


def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return sum(point_distance(points[i], points[i + 1]) for i in range(len(points) - 1))


def is_point_visible(px, py, screen_size, margin=20):
    return -margin <= px <= screen_size[0] + margin and -margin <= py <= screen_size[1] + margin


def simplify_label(text, max_len=24):
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def spaced_out(candidate, accepted, min_dist):
    for p in accepted:
        if point_distance(candidate, p) < min_dist:
            return False
    return True


# =========================
# 地图加载器
# =========================
class MapLoader:
    def __init__(self):
        self.graph = None           # coarse graph for RL
        self.full_graph = None      # full graph for visualization
        self.pois = None            # projected GeoDataFrame
        self.is_loading = True
        self.status_text = "Init..."

    def _get_largest_strongly_connected_component(self, G):
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        return G.subgraph(largest_scc).copy()

    def download_map(self):
        try:
            ox.settings.use_cache = True
            ox.settings.log_console = False
            os.makedirs("cache", exist_ok=True)

            # 1) 下载完整一点的 drive 路网
            self.status_text = "Downloading road network..."
            G = ox.graph_from_point(
                TARGET_COORDS,
                dist=VIEW_RADIUS + 500,
                network_type="drive",
                simplify=True,
            )
            print(f"[Step 1] 原始下载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

            # 2) 取最大强连通子图
            self.status_text = "Extracting connected component..."
            G = self._get_largest_strongly_connected_component(G)
            print(f"[Step 2] 强连通图后: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

            # 3) 投影 full graph
            self.status_text = "Projecting graph..."
            G_full = ox.project_graph(G)
            G_full = self._get_largest_strongly_connected_component(G_full)
            self.full_graph = G_full
            ox.save_graphml(G_full, "cache/taxi_graph_full.graphml")
            print(f"[Step 3] full graph 保存完成: {G_full.number_of_nodes()} 节点, {G_full.number_of_edges()} 边")

            # 4) 合并路口得到 coarse graph
            self.status_text = "Consolidating intersections..."
            G_coarse = ox.consolidate_intersections(
                G_full,
                tolerance=INTERSECTION_TOLERANCE,
                rebuild_graph=True,
                dead_ends=False,
            )
            G_coarse = self._get_largest_strongly_connected_component(G_coarse)
            self.graph = G_coarse
            ox.save_graphml(G_coarse, "cache/taxi_graph.graphml")
            print(f"[Step 4] coarse graph 保存完成: {G_coarse.number_of_nodes()} 节点, {G_coarse.number_of_edges()} 边")

            # 5) 下载 POI / 地标 / building
            self.status_text = "Downloading POIs/features..."
            tags = {
                "amenity": True,
                "shop": True,
                "tourism": True,
                "building": True,
                "office": True,
                "public_transport": True,
                "railway": True,
                "leisure": True,
                "landuse": True,
            }

            pois_gdf = ox.features_from_point(
                TARGET_COORDS,
                dist=VIEW_RADIUS,
                tags=tags,
            )

            if pois_gdf is not None and len(pois_gdf) > 0:
                # 不再用 ox.project_gdf(...)
                pois_gdf = pois_gdf.to_crs(G_full.graph["crs"])

                if "name" in pois_gdf.columns:
                    pois_gdf = pois_gdf[pois_gdf["name"].notna()].copy()

                self.pois = pois_gdf
                print(f"[Step 5] POI 下载完成: {len(pois_gdf)} 条")
            else:
                self.pois = None
                print("[Step 5] 未下载到 POI")

            self.status_text = "Done!"
            self.is_loading = False

            print("[Done] 已保存:")
            print("  - cache/taxi_graph_full.graphml")
            print("  - cache/taxi_graph.graphml")

        except Exception as e:
            self.status_text = f"Error: {e}"
            self.is_loading = False
            print(f"Loading Error: {e}")


# =========================
# 主程序
# =========================
def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("HKUST-GZ RL Environment Map")
    clock = pygame.time.Clock()

    font = load_label_font(18)
    small_font = load_label_font(14)
    tiny_font = load_label_font(12)

    loader = MapLoader()
    threading.Thread(target=loader.download_map, daemon=True).start()

    map_surface = None
    bounds = None
    angle = 0

    while True:
        screen.fill(BG_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if loader.is_loading:
            angle += 5
            rect = pygame.Rect(SCREEN_SIZE[0] // 2 - 30, SCREEN_SIZE[1] // 2 - 30, 60, 60)
            pygame.draw.arc(screen, ACCENT_COLOR, rect, math.radians(angle), math.radians(angle + 120), 3)
            txt = font.render(loader.status_text, True, (170, 170, 170))
            screen.blit(txt, (SCREEN_SIZE[0] // 2 - txt.get_width() // 2, SCREEN_SIZE[1] // 2 + 50))

        else:
            if map_surface is None and loader.graph is not None:
                # ========== 1. 计算可视区域 ==========
                xs = [data["x"] for _, data in loader.graph.nodes(data=True)]
                ys = [data["y"] for _, data in loader.graph.nodes(data=True)]

                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                span_x = max_x - min_x
                span_y = max_y - min_y
                max_span = max(span_x, span_y)

                cx = (min_x + max_x) / 2
                cy = (min_y + max_y) / 2

                bounds = (
                    cx - max_span / 2,
                    cx + max_span / 2,
                    cy - max_span / 2,
                    cy + max_span / 2,
                )

                map_surface = pygame.Surface(SCREEN_SIZE)
                map_surface.fill(BG_COLOR)

                # ========== 2. 画完整道路（full_graph 更真实） ==========
                road_label_points = []
                road_labels_drawn = 0

                draw_graph = loader.full_graph if loader.full_graph is not None else loader.graph

                for u, v, data in draw_graph.edges(data=True):
                    hw = get_edge_highway(data)
                    color = ROAD_COLORS.get(hw, ROAD_COLORS["default"])
                    width = 3 if hw in ["motorway", "trunk", "primary"] else 2 if hw in ["secondary", "tertiary"] else 1

                    if "geometry" in data and data["geometry"] is not None:
                        points = [
                            get_screen_coords(x, y, bounds, SCREEN_SIZE)
                            for x, y in data["geometry"].coords
                        ]
                    else:
                        u_node = draw_graph.nodes[u]
                        v_node = draw_graph.nodes[v]
                        points = [
                            get_screen_coords(u_node["x"], u_node["y"], bounds, SCREEN_SIZE),
                            get_screen_coords(v_node["x"], v_node["y"], bounds, SCREEN_SIZE),
                        ]

                    if len(points) >= 2:
                        pygame.draw.lines(map_surface, color, False, points, width)

                        # 绘制道路名称
                        if SHOW_ROAD_LABELS and road_labels_drawn < MAX_ROAD_LABELS:
                            road_name = get_edge_name(data)
                            if road_name:
                                mid = polyline_midpoint(points)
                                plen = polyline_length(points)
                                if (
                                    mid is not None
                                    and plen >= MIN_ROAD_LABEL_LENGTH
                                    and is_point_visible(mid[0], mid[1], SCREEN_SIZE)
                                    and spaced_out(mid, road_label_points, 55)
                                ):
                                    label = tiny_font.render(simplify_label(road_name, 20), True, (160, 160, 160))
                                    map_surface.blit(label, (mid[0] + 4, mid[1] + 2))
                                    road_label_points.append(mid)
                                    road_labels_drawn += 1

                # ========== 3. 画 coarse hub 节点 ==========
                for node_id, data in loader.graph.nodes(data=True):
                    px, py = get_screen_coords(data["x"], data["y"], bounds, SCREEN_SIZE)
                    if is_point_visible(px, py, SCREEN_SIZE, margin=0):
                        pygame.draw.circle(map_surface, (255, 255, 255), (px, py), 4)

                        name = safe_first(data.get("name")) or safe_first(data.get("ref"))
                        if SHOW_NAMED_NODES and name:
                            lbl = small_font.render(simplify_label(name, 18), True, (225, 225, 225))
                            map_surface.blit(lbl, (px + 8, py + 3))

                # ========== 4. 画 POI ==========
                if SHOW_POIS and loader.pois is not None and len(loader.pois) > 0:
                    poi_points = []
                    poi_count = 0

                    # 简单按“有 name 的、常见重要类型”优先
                    priority_rows = []
                    normal_rows = []

                    for _, row in loader.pois.iterrows():
                        name = row.get("name")
                        if not name:
                            continue

                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue

                        # Point / Polygon / MultiPolygon 都转成代表点
                        rep = geom.representative_point()
                        record = (row, rep)

                        amenity = safe_first(row.get("amenity"))
                        building = safe_first(row.get("building"))
                        tourism = safe_first(row.get("tourism"))

                        if amenity in {"university", "school", "bus_station"} or tourism or building:
                            priority_rows.append(record)
                        else:
                            normal_rows.append(record)

                    for row, rep in priority_rows + normal_rows:
                        if poi_count >= MAX_POI_LABELS:
                            break

                        px, py = get_screen_coords(rep.x, rep.y, bounds, SCREEN_SIZE)
                        if not is_point_visible(px, py, SCREEN_SIZE):
                            continue
                        if not spaced_out((px, py), poi_points, POI_MIN_PIXEL_DISTANCE):
                            continue

                        color = choose_poi_color(row)
                        pygame.draw.circle(map_surface, color, (px, py), 4)

                        if SHOW_POI_LABELS:
                            name = simplify_label(row.get("name"), 22)
                            lbl = small_font.render(name, True, color)
                            map_surface.blit(lbl, (px + 6, py - 2))

                        poi_points.append((px, py))
                        poi_count += 1

            # 显示缓存好的地图
            if map_surface is not None:
                screen.blit(map_surface, (0, 0))

                # 图例
                title = font.render("Legend", True, (230, 230, 230))
                screen.blit(title, (20, 18))

                y0 = 50
                for i, (type_name, color) in enumerate(ROAD_COLORS.items()):
                    pygame.draw.rect(screen, color, (20, y0 + i * 20, 15, 10))
                    lbl = tiny_font.render(type_name, True, (200, 200, 200))
                    screen.blit(lbl, (45, y0 - 3 + i * 20))

                y1 = y0 + len(ROAD_COLORS) * 20 + 15
                pygame.draw.circle(screen, (255, 255, 255), (28, y1), 4)
                screen.blit(tiny_font.render("RL hubs", True, (210, 210, 210)), (45, y1 - 8))

                y2 = y1 + 22
                pygame.draw.circle(screen, POI_COLORS["default"], (28, y2), 4)
                screen.blit(tiny_font.render("POIs / landmarks", True, (210, 210, 210)), (45, y2 - 8))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()