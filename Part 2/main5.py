import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import math

# --- Helper Geometric Functions ---

def subtract_points(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1])

def add_points(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])

def scale_point(p, scalar):
    return (p[0] * scalar, p[1] * scalar)

def cross_product_2d(p1, p2):
    """Calculates the 2D cross product (z-component of 3D cross product)."""
    return p1[0] * p2[1] - p1[1] * p2[0]

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def distance_sq(p1, p2):
    """Squared distance to avoid sqrt for comparisons."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def distance(p1, p2):
    return math.sqrt(distance_sq(p1, p2))

def normalize_vector(v):
    mag = math.sqrt(v[0]**2 + v[1]**2)
    if mag == 0:
        return (0, 0)
    return (v[0] / mag, v[1] / mag)

# --- Core Algorithm Implementations ---

def check_polygon_convexity(polygon_vertices):
    """
    Checks if a polygon is convex.
    Assumes vertices are ordered (CW or CCW).
    Returns: "convex_ccw", "convex_cw", "concave", "degenerate"
    """
    if len(polygon_vertices) < 3:
        return "degenerate" # Not a polygon

    got_positive = False
    got_negative = False
    
    num_vertices = len(polygon_vertices)
    for i in range(num_vertices):
        p1 = polygon_vertices[i]
        p2 = polygon_vertices[(i + 1) % num_vertices]
        p3 = polygon_vertices[(i + 2) % num_vertices]

        vec1 = subtract_points(p2, p1)
        vec2 = subtract_points(p3, p2)
        
        # Cross product: vec1 x vec2
        cp = cross_product_2d(vec1, vec2)

        if cp > 1e-9: # Using a small epsilon for float comparison
            got_positive = True
        elif cp < -1e-9:
            got_negative = True
        # If cp is close to 0, points are collinear, can be part of a convex shape

        if got_positive and got_negative:
            return "concave"

    if got_positive:
        return "convex_ccw" # Conventionally, positive cross product means CCW turn
    elif got_negative:
        return "convex_cw"  # Negative cross product means CW turn
    else:
        return "degenerate" # All points are collinear

def get_internal_normals(polygon_vertices, convexity_type):
    """
    Calculates internal normals for a convex polygon.
    convexity_type should be "convex_ccw" or "convex_cw".
    Returns a list of normals, one for each edge. Each normal is ((x,y), (mid_x, mid_y))
    where (x,y) is the normal vector and (mid_x, mid_y) is the midpoint of the edge.
    """
    if not (convexity_type == "convex_ccw" or convexity_type == "convex_cw"):
        return [] # Not convex or unknown winding

    normals = []
    num_vertices = len(polygon_vertices)
    for i in range(num_vertices):
        p1 = polygon_vertices[i]
        p2 = polygon_vertices[(i + 1) % num_vertices]

        edge_vector = subtract_points(p2, p1)
        
        # Perpendicular vector
        # For CCW: normal is (-dy, dx) to point inward
        # For CW: normal is (dy, -dx) to point inward
        if convexity_type == "convex_ccw":
            normal_vec = (-edge_vector[1], edge_vector[0])
        else: # convex_cw
            normal_vec = (edge_vector[1], -edge_vector[0])
        
        normalized_normal = normalize_vector(normal_vec)
        
        mid_point = ( (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 )
        normals.append({'vector': normalized_normal, 'origin': mid_point, 'edge': (p1,p2)})
        
    return normals
    
def graham_scan(points):
    """
    Computes the convex hull of a set of points using Graham Scan.
    Returns a list of points forming the convex hull, ordered counter-clockwise.
    """
    if len(points) < 3:
        return points # Convex hull is the points themselves

    # Find P0: point with the lowest y-coordinate, then lowest x-coordinate
    p0 = min(points, key=lambda p: (p[1], p[0]))

    # Sort points by polar angle with P0
    # If angles are the same, sort by distance to P0 (closer first)
    def polar_angle_and_dist_sq(p):
        if p == p0:
            return -float('inf'), 0 # p0 first
        angle = math.atan2(p[1] - p0[1], p[0] - p0[0])
        return angle, distance_sq(p0, p)

    sorted_points = sorted(points, key=polar_angle_and_dist_sq)
    
    # Filter out collinear points that are closer to p0 than another collinear point
    # This step is crucial for robustness if multiple points have the same angle.
    # We keep the farthest point among those with the same angle.
    filtered_points = []
    if sorted_points:
        filtered_points.append(sorted_points[0]) # p0
        for i in range(1, len(sorted_points)):
            # If angle is same as previous, only add if it's different (farthest was chosen by sort)
            # Or if current point is not p0 and previous was.
            # A simple way: remove points that form a zero area triangle with p0 and next point
            # if they are closer than the next point.
            # For now, the sort should handle it by distance for same angle,
            # but we need to handle strictly collinear points carefully.
            # A robust way is to remove points P_i if P0, P_i, P_{i+1} are collinear and P_i is between P0 and P_{i+1}.
            # The provided sort by distance for same angle helps. Let's ensure no duplicates.
            if len(filtered_points) == 1 or polar_angle_and_dist_sq(sorted_points[i])[0] != polar_angle_and_dist_sq(filtered_points[-1])[0]:
                 filtered_points.append(sorted_points[i])
            else: # Same angle, update if this one is further
                if distance_sq(p0, sorted_points[i]) > distance_sq(p0, filtered_points[-1]):
                    filtered_points[-1] = sorted_points[i]


    if len(filtered_points) < 3: # After filtering, not enough unique angle points
        return filtered_points


    hull = []
    for p_i in filtered_points:
        # While last two points in hull and p_i make a non-left turn (or are collinear)
        while len(hull) >= 2:
            p_k = hull[-2] # second to last
            p_j = hull[-1] # last
            # cross_product(p_j - p_k, p_i - p_j)
            # If > 0, left turn. If < 0, right turn. If == 0, collinear.
            # We want left turns for CCW hull.
            # (pj.x - pk.x) * (pi.y - pj.y) - (pj.y - pk.y) * (pi.x - pj.x)
            val = cross_product_2d(subtract_points(p_j, p_k), subtract_points(p_i, p_j))
            if val > 1e-9: # Left turn
                break
            hull.pop() # Remove p_j, it's a right turn or collinear inside
        hull.append(p_i)
    return hull

def jarvis_march(points):
    """
    Computes the convex hull of a set of points using Jarvis March (Gift Wrapping).
    Returns a list of points forming the convex hull, ordered counter-clockwise.
    """
    if len(points) < 3:
        return points

    # Find the point with the lowest y-coordinate (leftmost if tied) - starting point
    start_point = min(points, key=lambda p: (p[1], p[0]))

    hull = []
    current_point = start_point
    
    while True:
        hull.append(current_point)
        next_point = points[0]
        if next_point == current_point: # Ensure next_point is initially different
             if len(points) > 1: next_point = points[1]
             else: break # Only one point

        for candidate_point in points:
            if candidate_point == current_point:
                continue
            
            # Orientation: cross_product(next_point - current_point, candidate_point - current_point)
            # If > 0, candidate_point is to the left of current_point -> next_point vector
            # If < 0, candidate_point is to the right
            # If == 0, candidate_point is collinear
            vec_curr_next = subtract_points(next_point, current_point)
            vec_curr_cand = subtract_points(candidate_point, current_point)
            cp = cross_product_2d(vec_curr_next, vec_curr_cand)

            if cp < -1e-9: # candidate_point is to the "right" of current->next_point, making it more CCW
                next_point = candidate_point
            elif abs(cp) < 1e-9: # Collinear
                # If candidate_point is further along the line than next_point
                if distance_sq(current_point, candidate_point) > distance_sq(current_point, next_point):
                    next_point = candidate_point
        
        current_point = next_point
        if current_point == start_point:
            break
        if len(hull) > len(points) : # Safety break, something went wrong
            print("Jarvis March safety break")
            return hull # return what we have
    return hull

def line_segment_intersects_polygon(seg_p1, seg_p2, polygon_vertices):
    """
    Finds intersection points of a line segment with polygon edges.
    Returns a list of intersection points.
    """
    intersections = []
    num_vertices = len(polygon_vertices)
    if num_vertices < 2: return []

    for i in range(num_vertices):
        poly_p1 = polygon_vertices[i]
        poly_p2 = polygon_vertices[(i + 1) % num_vertices]

        # Line segment P1P2: P1 + t(P2-P1) for 0<=t<=1
        # Polygon edge   A B: A  + u(B -A) for 0<=u<=1
        
        # Let P1=(x1,y1), P2=(x2,y2) for segment
        # Let A=(x3,y3), B=(x4,y4) for polygon edge
        x1, y1 = seg_p1
        x2, y2 = seg_p2
        x3, y3 = poly_p1
        x4, y4 = poly_p2

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(den) < 1e-9: # Lines are parallel or collinear
            # Check for collinear overlapping segments (more complex, skipping for now for simplicity)
            # For this lab, we assume non-collinear distinct intersections primarily
            continue

        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) # Corrected u_num derivation

        t = t_num / den
        u = u_num / den

        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection point
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            intersections.append((ix, iy))
            
    return list(set(intersections)) # Remove duplicates

def is_point_in_polygon(point, polygon_vertices):
    """
    Checks if a point is inside a polygon using the ray casting algorithm.
    Returns True if inside, False otherwise.
    Handles points on boundary as inside for simplicity here, can be refined.
    """
    if not polygon_vertices or len(polygon_vertices) < 3:
        return False

    px, py = point
    num_vertices = len(polygon_vertices)
    intersections = 0

    for i in range(num_vertices):
        p1 = polygon_vertices[i]
        p2 = polygon_vertices[(i + 1) % num_vertices]

        p1x, p1y = p1
        p2x, p2y = p2

        # Check if point is on a vertex
        if (px == p1x and py == p1y) or (px == p2x and py == p2y):
            return True # On boundary (vertex)

        # Check if point is on a horizontal segment
        if p1y == p2y == py and min(p1x, p2x) <= px <= max(p1x, p2x):
            return True # On boundary (horizontal edge)
        
        # Check if point is on a vertical segment
        if p1x == p2x == px and min(p1y, p2y) <= py <= max(p1y, p2y):
            return True # On boundary (vertical edge)

        # Ray casting: (px, py) to (infinity, py)
        # Check if the edge (p1,p2) crosses the horizontal ray
        if (p1y <= py < p2y or p2y <= py < p1y): # Point y is between edge's y-range
            # Compute x-intersection of ray with line extending edge
            # (py - p1y) / (p2y - p1y) = (vt - p1x) / (p2x - p1x)
            # vt = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if abs(p2y - p1y) > 1e-9: # Avoid division by zero for horizontal edge (already handled)
                x_intersection = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if x_intersection > px: # Intersection is to the right of the point
                    intersections += 1
    
    return intersections % 2 == 1


# --- Tkinter Application ---
class GraphicsEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Простой Графический Редактор")
        self.root.geometry("1000x800")

        self.current_tool = "none"
        self.temp_points = []
        self.polygons = []
        self.lines = []
        self.convex_hulls = []
        self.intersection_points_viz = []
        self.normals_viz = []

        self.selected_polygon_idx = None
        self.selected_point_for_test = None
        self.selected_line_for_test = None

        # --- Menu ---
        menubar = tk.Menu(root)
        
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Очистить холст", command=self.clear_canvas)
        filemenu.add_separator()
        filemenu.add_command(label="Выход", command=root.quit)
        menubar.add_cascade(label="Файл", menu=filemenu)

        toolsmenu = tk.Menu(menubar, tearoff=0)
        toolsmenu.add_command(label="Рисовать полигон", command=lambda: self.set_tool("draw_polygon"))
        toolsmenu.add_command(label="Рисовать линию", command=lambda: self.set_tool("draw_line"))
        toolsmenu.add_separator()
        toolsmenu.add_command(label="Проверить выпуклость (последний полигон)", command=self.check_last_polygon_convexity)
        toolsmenu.add_command(label="Найти внутренние нормали (последний полигон)", command=self.find_last_polygon_normals)
        toolsmenu.add_separator()
        toolsmenu.add_command(label="Построить выпуклую оболочку (Грэхем, все точки)", command=lambda: self.build_convex_hull("graham"))
        toolsmenu.add_command(label="Построить выпуклую оболочку (Джарвис, все точки)", command=lambda: self.build_convex_hull("jarvis"))
        toolsmenu.add_separator()
        toolsmenu.add_command(label="Пересечение отрезка с полигоном", command=self.setup_line_polygon_intersection)
        toolsmenu.add_command(label="Принадлежность точки полигону", command=self.setup_point_in_polygon)
        
        menubar.add_cascade(label="Инструменты", menu=toolsmenu)
        root.config(menu=menubar)

        # --- Toolbar (Simplified as buttons for now) ---
        toolbar = ttk.Frame(root, padding="5")
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Полигон", command=lambda: self.set_tool("draw_polygon")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Линия", command=lambda: self.set_tool("draw_line")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Очистить", command=self.clear_canvas).pack(side=tk.LEFT, padx=2)


        # --- Canvas ---
        self.canvas = tk.Canvas(root, bg="white", width=800, height=600)
        self.canvas.pack(pady=20, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click_left)
        self.canvas.bind("<Button-3>", self.on_canvas_click_right)

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_var.set("Готов. Выберите инструмент.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.draw_all()

    def set_tool(self, tool_name):
        self.current_tool = tool_name
        self.temp_points = []
        self.selected_point_for_test = None
        self.selected_line_for_test = None
        self.status_var.set(f"Инструмент: {tool_name}. Кликните на холсте.")
        if tool_name == "point_in_poly_select_point":
            self.status_var.set("Кликните, чтобы выбрать точку для проверки.")
        elif tool_name == "line_intersect_select_line":
             self.status_var.set("Нарисуйте отрезок для проверки (2 клика).")
        self.draw_all()


    def on_canvas_click_left(self, event):
        x, y = event.x, event.y
        
        if self.current_tool == "draw_polygon":
            self.temp_points.append((x, y))
            self.status_var.set(f"Добавлена вершина {len(self.temp_points)} для полигона. Правый клик для завершения.")
        elif self.current_tool == "draw_line":
            self.temp_points.append((x,y))
            if len(self.temp_points) == 2:
                self.lines.append(list(self.temp_points))
                self.status_var.set(f"Линия нарисована. {self.temp_points[0]} -> {self.temp_points[1]}")
                self.temp_points = []
        elif self.current_tool == "point_in_poly_select_point":
            self.selected_point_for_test = (x,y)
            self.status_var.set(f"Точка ({x},{y}) выбрана. Теперь выберите полигон (клик внутри).")
            self.current_tool = "point_in_poly_select_polygon"
        elif self.current_tool == "point_in_poly_select_polygon":
            if self.selected_point_for_test and self.polygons:
                target_polygon = self.polygons[-1]
                is_inside = is_point_in_polygon(self.selected_point_for_test, target_polygon)
                result_text = "внутри" if is_inside else "снаружи"
                messagebox.showinfo("Результат: Точка в полигоне", 
                                    f"Точка {self.selected_point_for_test} находится {result_text} полигона {target_polygon}.")
                self.status_var.set(f"Точка {self.selected_point_for_test} {result_text} полигона. Выберите новый инструмент.")
            else:
                self.status_var.set("Сначала выберите точку, затем убедитесь, что полигон нарисован.")
            self.current_tool = "none"
            self.selected_point_for_test = None
        elif self.current_tool == "line_intersect_select_line":
            self.temp_points.append((x,y))
            if len(self.temp_points) == 2:
                self.selected_line_for_test = list(self.temp_points)
                self.temp_points = []
                if self.polygons:
                    target_polygon = self.polygons[-1]
                    intersections = line_segment_intersects_polygon(
                        self.selected_line_for_test[0],
                        self.selected_line_for_test[1],
                        target_polygon
                    )
                    self.intersection_points_viz = intersections
                    if intersections:
                        messagebox.showinfo("Пересечение отрезка с полигоном",
                                            f"Найдены пересечения: {intersections}")
                        self.status_var.set(f"Найдены пересечения: {len(intersections)}. См. на холсте.")
                    else:
                        messagebox.showinfo("Пересечение отрезка с полигоном", "Пересечений не найдено.")
                        self.status_var.set("Пересечений не найдено.")
                else:
                    messagebox.showwarning("Ошибка", "Сначала нарисуйте полигон для проверки пересечения.")
                self.current_tool = "none"
            else:
                self.status_var.set("Отрезок: кликните для второй точки.")


        self.draw_all()

    def on_canvas_click_right(self, event):
        if self.current_tool == "draw_polygon" and len(self.temp_points) >= 3:
            self.polygons.append(list(self.temp_points))
            self.status_var.set(f"Полигон с {len(self.temp_points)} вершинами нарисован.")
            self.temp_points = []
            self.selected_polygon_idx = len(self.polygons) - 1
        elif self.current_tool == "draw_polygon":
             self.status_var.set("Нужно как минимум 3 вершины для полигона.")
        self.draw_all()

    def draw_all(self):
        self.canvas.delete("all")
        
        # Draw current temp points for polygon/line
        if self.temp_points:
            if self.current_tool == "draw_polygon":
                if len(self.temp_points) > 1:
                    self.canvas.create_line(self.temp_points, fill="gray", dash=(2,2))
                for p in self.temp_points:
                    self.canvas.create_oval(p[0]-3, p[1]-3, p[0]+3, p[1]+3, fill="blue", outline="blue")
            elif self.current_tool == "draw_line" or self.current_tool == "line_intersect_select_line":
                 if len(self.temp_points) == 1:
                    p = self.temp_points[0]
                    self.canvas.create_oval(p[0]-3, p[1]-3, p[0]+3, p[1]+3, fill="green", outline="green")

        # Draw polygons
        for i, poly in enumerate(self.polygons):
            color = "black"
            if self.selected_polygon_idx == i: color = "red"
            if len(poly) > 1 : self.canvas.create_polygon(poly, outline=color, fill="", width=2)
            for p_idx, p_vertex in enumerate(poly):
                 self.canvas.create_oval(p_vertex[0]-2, p_vertex[1]-2, p_vertex[0]+2, p_vertex[1]+2, fill=color)



        # Draw lines
        for line in self.lines:
            self.canvas.create_line(line, fill="purple", width=2)
        
        # Draw selected line for intersection test
        if self.selected_line_for_test:
            self.canvas.create_line(self.selected_line_for_test, fill="orange", width=3, dash=(4,2))

        # Draw convex hulls
        for hull in self.convex_hulls:
            if len(hull) > 1:
                 self.canvas.create_polygon(hull, outline="green", fill="", width=3, dash=(4,4))
            for p in hull:
                 self.canvas.create_oval(p[0]-4, p[1]-4, p[0]+4, p[1]+4, fill="green", outline="darkgreen")

        # Draw intersection points
        for p_inter in self.intersection_points_viz:
            self.canvas.create_oval(p_inter[0]-4, p_inter[1]-4, p_inter[0]+4, p_inter[1]+4, fill="red", outline="darkred")
            self.canvas.create_text(p_inter[0], p_inter[1]-10, text="Intersection", fill="red")

        # Draw normals
        for normal_data in self.normals_viz:
            orig = normal_data['origin']
            vec = normal_data['vector']
            # Scale normal for visibility
            end_point = (orig[0] + vec[0] * 20, orig[1] + vec[1] * 20)
            self.canvas.create_line(orig, end_point, fill="cyan", arrow=tk.LAST, width=2)

        # Draw selected point for test
        if self.selected_point_for_test:
            p = self.selected_point_for_test
            self.canvas.create_oval(p[0]-5, p[1]-5, p[0]+5, p[1]+5, fill="magenta", outline="magenta")
            self.canvas.create_text(p[0], p[1]-10, text="Test Point", fill="magenta")


    def clear_canvas(self):
        self.temp_points = []
        self.polygons = []
        self.lines = []
        self.convex_hulls = []
        self.intersection_points_viz = []
        self.normals_viz = []
        self.selected_polygon_idx = None
        self.selected_point_for_test = None
        self.selected_line_for_test = None
        self.current_tool = "none"
        self.status_var.set("Холст очищен. Готов.")
        self.draw_all()

    def check_last_polygon_convexity(self):
        if not self.polygons:
            messagebox.showwarning("Ошибка", "Сначала нарисуйте полигон.")
            return

        last_poly = self.polygons[-1]
        if len(last_poly) < 3:
            messagebox.showinfo("Результат выпуклости", "Недостаточно вершин для полигона.")
            return
            
        convexity = check_polygon_convexity(last_poly)
        
        msg = f"Полигон: {convexity.replace('_', ' ')}"
        if convexity == "convex_ccw": msg = "Выпуклый (обход против часовой стрелки)"
        elif convexity == "convex_cw": msg = "Выпуклый (обход по часовой стрелке)"
        elif convexity == "concave": msg = "Вогнутый"
        elif convexity == "degenerate": msg = "Вырожденный (вершины коллинеарны)"
        
        messagebox.showinfo("Результат выпуклости", msg)
        self.status_var.set(f"Проверка выпуклости: {msg}")

    def find_last_polygon_normals(self):
        if not self.polygons:
            messagebox.showwarning("Ошибка", "Сначала нарисуйте полигон.")
            return

        last_poly = self.polygons[-1]
        convexity = check_polygon_convexity(last_poly)

        if convexity == "concave" or convexity == "degenerate":
            messagebox.showwarning("Нормали", "Нормали можно найти только для выпуклых полигонов.")
            self.normals_viz = []
        else:
            self.normals_viz = get_internal_normals(last_poly, convexity)
            self.status_var.set(f"Внутренние нормали для полигона найдены ({len(self.normals_viz)}).")
        self.draw_all()


    def build_convex_hull(self, method):
        all_points = []
        for poly in self.polygons:
            all_points.extend(poly)
        for line in self.lines:
            all_points.extend(line)

        if not all_points:
            messagebox.showwarning("Ошибка", "На холсте нет точек для построения оболочки.")
            return
        
        unique_points = sorted(list(set(map(tuple, all_points))))

        if len(unique_points) < 3:
            messagebox.showinfo("Выпуклая оболочка", "Нужно как минимум 3 уникальные точки для оболочки.")
            self.convex_hulls = [unique_points] if unique_points else []
        else:
            hull_points = []
            if method == "graham":
                hull_points = graham_scan(unique_points)
                self.status_var.set(f"Выпуклая оболочка (Грэхем) построена: {len(hull_points)} вершин.")
            elif method == "jarvis":
                hull_points = jarvis_march(unique_points)
                self.status_var.set(f"Выпуклая оболочка (Джарвис) построена: {len(hull_points)} вершин.")
            
            self.convex_hulls = [hull_points] 
        
        self.draw_all()

    def setup_point_in_polygon(self):
        if not self.polygons:
            messagebox.showwarning("Ошибка", "Сначала нарисуйте хотя бы один полигон.")
            return
        self.set_tool("point_in_poly_select_point")

    def setup_line_polygon_intersection(self):
        if not self.polygons:
            messagebox.showwarning("Ошибка", "Сначала нарисуйте хотя бы один полигон.")
            return
        self.set_tool("line_intersect_select_line")


if __name__ == "__main__":
    root = tk.Tk()
    app = GraphicsEditorApp(root)
    root.mainloop()