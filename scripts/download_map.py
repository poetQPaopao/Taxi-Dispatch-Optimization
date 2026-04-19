import pygame
import osmnx as ox
import networkx as nx
import threading
import sys
import os

# --- 配置参数 ---
# 目标坐标（港科广红鸟广场附近的真实 WGS-84 坐标）
TARGET_COORDS = (22.894, 113.478)
# 视界半径，1500米既能覆盖校园周边，又不会导致投影死机
VIEW_RADIUS = 3000
# 路口合并容差（米），15米是城市路网抽象的黄金数值
INTERSECTION_TOLERANCE = 20.0
SCREEN_SIZE = (1000, 800)
BG_COLOR = (15, 16, 18)
ACCENT_COLOR = (0, 200, 255) 
SHOW_NAMED_NODES = True

ROAD_COLORS = {
    'motorway': (255, 90, 90),
    'trunk': (255, 160, 0),
    'primary': (255, 210, 50),
    'secondary': (180, 255, 50),
    'tertiary': (80, 220, 255),
    'residential': (130, 130, 140),
    'service': (80, 80, 100),
    'unclassified': (60, 60, 60),
    'default': (70, 70, 75)
}

class MapLoader:
    def __init__(self):
        self.graph = None
        self.is_loading = True
        self.status_text = "Init..."

    def _get_largest_strongly_connected_component(self, G):
        """
        使用原生的 networkx 方法，彻底解决 OSMnx 版本兼容性问题
        """
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        return G.subgraph(largest_scc).copy()

    def download_map(self):
        try:
            self.status_text = "Downloading HKUST-GZ Map..."
            ox.settings.use_cache = True
            
            # --- 1. 下载并过滤路网 ---
            G = ox.graph_from_point(
                TARGET_COORDS,
                dist=VIEW_RADIUS + 500,
                custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified"]',
                simplify=True,
            )
            print(f"[Step 1] 原始下载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

            # --- 2. 提取连通图 ---
            G = self._get_largest_strongly_connected_component(G)
            nodes_count = G.number_of_nodes()
            print(f"[Step 2] 提取连通图后: {nodes_count} 节点, {G.number_of_edges()} 边")

            # --- 3. 投影 (耗时步骤) ---
            self.status_text = f"Projecting {nodes_count} nodes..."

            G = ox.project_graph(G)
            
            # --- 4. 合并密集路口 (降维核心) ---
            self.status_text = "Consolidating Intersections..."
            G = ox.consolidate_intersections(
                G,
                tolerance=INTERSECTION_TOLERANCE, 
                rebuild_graph=True,
                dead_ends=False,
            )
            print(f"[Step 3] 合并路口后: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

            # --- 5. 最终清理 ---
            G = self._get_largest_strongly_connected_component(G)
            print(f"[Step 4] 最终确定 Hubs: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

            self.graph = G
            self.is_loading = False
            self.status_text = "Done!"
        except Exception as e:
            self.status_text = f"Error: {e}"
            print(f"Loading Error: {e}")

def get_screen_coords(x, y, bounds, screen_size):
    min_x, max_x, min_y, max_y = bounds
    w, h = screen_size
    padding = 50
    px = padding + (x - min_x) / (max_x - min_x) * (w - 2 * padding)
    py = h - (padding + (y - min_y) / (max_y - min_y) * (h - 2 * padding))
    return int(px), int(py)

def load_label_font(size):
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf"
    ]
    for path in candidates:
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    for name in ["PingFang SC", "Heiti SC", "Arial Unicode MS"]:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)

def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("HKUST-GZ RL Environment Map")
    clock = pygame.time.Clock()
    font = load_label_font(18)

    loader = MapLoader()
    threading.Thread(target=loader.download_map, daemon=True).start()

    map_surface = None
    bounds = None
    angle = 0

    while True:
        screen.fill(BG_COLOR)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        if loader.is_loading:
            # 绘制加载动画
            angle += 5
            rect = pygame.Rect(SCREEN_SIZE[0]//2-30, SCREEN_SIZE[1]//2-30, 60, 60)
            pygame.draw.arc(screen, ACCENT_COLOR, rect, angle*0.017, (angle+120)*0.017, 3)
            txt = font.render(loader.status_text, True, (150, 150, 150))
            screen.blit(txt, (SCREEN_SIZE[0]//2 - txt.get_width()//2, SCREEN_SIZE[1]//2 + 50))
        else:
            if map_surface is None and loader.graph is not None:
                # --- 1. 自动计算完美的屏幕视野 (Bounding Box) ---
                xs = [data['x'] for node, data in loader.graph.nodes(data=True)]
                ys = [data['y'] for node, data in loader.graph.nodes(data=True)]
                
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                span_x = max_x - min_x
                span_y = max_y - min_y
                max_span = max(span_x, span_y)
                
                cx = (min_x + max_x) / 2
                cy = (min_y + max_y) / 2
                bounds = (cx - max_span/2, cx + max_span/2, cy - max_span/2, cy + max_span/2)

                map_surface = pygame.Surface(SCREEN_SIZE)
                map_surface.fill(BG_COLOR)
                
                # --- 2. 绘制真实的弯曲道路 ---
                for u, v, data in loader.graph.edges(data=True):
                    hw = data.get('highway', 'unclassified')
                    if isinstance(hw, list): hw = hw[0]
                    color = ROAD_COLORS.get(hw, ROAD_COLORS['default'])
                    width = 2 if hw in ['primary', 'motorway', 'trunk'] else 1
                    
                    # 优先读取 geometry 绘制曲线
                    if 'geometry' in data:
                        points = [get_screen_coords(x, y, bounds, SCREEN_SIZE) for x, y in data['geometry'].coords]
                        if len(points) >= 2:
                            pygame.draw.lines(map_surface, color, False, points, width)
                    else:
                        u_node = loader.graph.nodes[u]
                        v_node = loader.graph.nodes[v]
                        p1 = get_screen_coords(u_node['x'], u_node['y'], bounds, SCREEN_SIZE)
                        p2 = get_screen_coords(v_node['x'], v_node['y'], bounds, SCREEN_SIZE)
                        pygame.draw.line(map_surface, color, p1, p2, width)

                # --- 3. 绘制最终的 RL Agent 枢纽节点 (Hubs) ---
                for node_id, data in loader.graph.nodes(data=True):
                    px, py = get_screen_coords(data['x'], data['y'], bounds, SCREEN_SIZE)
                    if 0 <= px < SCREEN_SIZE[0] and 0 <= py < SCREEN_SIZE[1]:
                        # 用明显的白色圆点标注 Hub
                        pygame.draw.circle(map_surface, (255, 255, 255), (px, py), 5)
                        
                        # 绘制路名 (如果有)
                        name = data.get('name') or data.get('ref')
                        if SHOW_NAMED_NODES and name:
                            # 处理列表类型的名字
                            if isinstance(name, list):
                                name = name[0]
                            lbl = font.render(str(name), True, (220, 220, 220))
                            map_surface.blit(lbl, (px + 8, py + 3))

            if map_surface:
                screen.blit(map_surface, (0, 0))
                
                # 绘制图例
                for i, (type_name, color) in enumerate(ROAD_COLORS.items()):
                    pygame.draw.rect(screen, color, (20, 20 + i*20, 15, 10))
                    lbl = font.render(type_name, True, (200, 200, 200))
                    screen.blit(lbl, (45, 15 + i*20))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()