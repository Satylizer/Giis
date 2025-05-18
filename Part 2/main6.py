import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math

# --- Constants ---
PIXEL_SIZE = 1  # For drawing individual pixels (can be > 1 for visibility)
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 500
BACKGROUND_COLOR = "white"
POLYGON_EDGE_COLOR = "black"
FILL_COLOR_SCANLINE = "lightblue"
FILL_COLOR_SEED = "lightgreen"
HIGHLIGHT_COLOR = "red" # For debug highlighting

# --- Helper: Bresenham's Line Algorithm (to get all points on an edge) ---
def bresenham_line(x1, y1, x2, y2):
    """Generates all points on a line from (x1, y1) to (x2, y2)"""
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

class PolygonFillerApp:
    def __init__(self, master):
        self.master = master
        master.title("Polygon Filling Algorithms")

        self.vertices = []
        self.current_polygon_edges = [] # Store drawn edge IDs
        self.boundary_pixels = set()
        self.filled_pixels_display = {} # Store (x,y) -> item_id for filled pixels

        self.selected_algorithm = tk.StringVar(value="scanline_ordered_edge_list")
        self.seed_point = None
        self.is_drawing_polygon = True
        self.debug_mode = tk.BooleanVar(value=False)
        self.debug_generator = None
        self.debug_step_count = 0

        # --- Main Panes ---
        main_pane = tk.PanedWindow(master, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # --- Left Pane (Controls) ---
        controls_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(controls_frame, stretch="never")

        # Algorithm Selection
        ttk.Label(controls_frame, text="Select Algorithm:").pack(pady=5)
        algorithms = [
            ("Scanline (Ordered Edge List)", "scanline_ordered_edge_list"),
            ("Scanline (Active Edge List)", "scanline_ael"),
            ("Simple Seed Fill (4-conn)", "seed_fill_simple"),
            ("Scanline Seed Fill (4-conn)", "seed_fill_scanline")
        ]
        for text, mode in algorithms:
            ttk.Radiobutton(controls_frame, text=text, variable=self.selected_algorithm, value=mode, command=self.reset_debug).pack(anchor=tk.W)

        # Buttons
        self.draw_button = ttk.Button(controls_frame, text="Draw Polygon (Click on Canvas)", command=self.start_drawing_polygon)
        self.draw_button.pack(fill=tk.X, pady=5)
        self.draw_button.config(state=tk.DISABLED) # Start in drawing mode

        self.finish_polygon_button = ttk.Button(controls_frame, text="Finish Polygon", command=self.finish_polygon)
        self.finish_polygon_button.pack(fill=tk.X, pady=5)
        self.finish_polygon_button.config(state=tk.DISABLED)

        self.fill_button = ttk.Button(controls_frame, text="Fill Polygon", command=self.start_fill)
        self.fill_button.pack(fill=tk.X, pady=5)
        self.fill_button.config(state=tk.DISABLED)

        ttk.Checkbutton(controls_frame, text="Debug Mode", variable=self.debug_mode, command=self.reset_debug).pack(pady=5)
        self.next_step_button = ttk.Button(controls_frame, text="Next Step", command=self.execute_next_debug_step)
        self.next_step_button.pack(fill=tk.X, pady=5)
        self.next_step_button.config(state=tk.DISABLED)

        self.clear_button = ttk.Button(controls_frame, text="Clear All", command=self.clear_all)
        self.clear_button.pack(fill=tk.X, pady=5)

        # Debug Info Area
        ttk.Label(controls_frame, text="Debug Information:").pack(pady=(10,0))
        self.debug_text_area = scrolledtext.ScrolledText(controls_frame, height=15, width=40, wrap=tk.WORD)
        self.debug_text_area.pack(fill=tk.BOTH, expand=True)
        self.debug_text_area.config(state=tk.DISABLED)

        # --- Right Pane (Canvas) ---
        self.canvas_frame = ttk.Frame(main_pane)
        main_pane.add(self.canvas_frame, stretch="always")

        self.canvas = tk.Canvas(self.canvas_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg=BACKGROUND_COLOR)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.handle_canvas_click)

        self.status_bar = ttk.Label(master, text="Mode: Drawing Polygon. Click to add vertices.", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _draw_pixel(self, x, y, color, tag="pixel"):
        """Draws a single 'pixel' on the canvas."""
        # Ensure pixel is within canvas bounds if PIXEL_SIZE > 1
        if 0 <= x < CANVAS_WIDTH and 0 <= y < CANVAS_HEIGHT:
            # Remove previously filled pixel at this location if any
            if (x,y) in self.filled_pixels_display:
                self.canvas.delete(self.filled_pixels_display[(x,y)])

            item_id = self.canvas.create_rectangle(
                x, y, x + PIXEL_SIZE, y + PIXEL_SIZE,
                fill=color, outline=color, tags=(tag,)
            )
            self.filled_pixels_display[(x,y)] = item_id
            return item_id
        return None

    def _draw_line_on_canvas(self, x1, y1, x2, y2, color=POLYGON_EDGE_COLOR):
        line_id = self.canvas.create_line(x1, y1, x2, y2, fill=color, width=PIXEL_SIZE) # Ensure lines are thin
        self.current_polygon_edges.append(line_id)
        # Add pixels of this line to boundary_pixels
        for px, py in bresenham_line(int(x1), int(y1), int(x2), int(y2)):
            self.boundary_pixels.add((px, py))

    def start_drawing_polygon(self):
        self.clear_all() # Clear everything before starting a new polygon
        self.is_drawing_polygon = True
        self.draw_button.config(state=tk.DISABLED)
        self.finish_polygon_button.config(state=tk.NORMAL)
        self.fill_button.config(state=tk.DISABLED)
        self.status_bar.config(text="Mode: Drawing Polygon. Click to add vertices.")

    def handle_canvas_click(self, event):
        x, y = event.x, event.y
        # Snap to a pseudo-grid if PIXEL_SIZE > 1, though not strictly necessary for 1
        x = (x // PIXEL_SIZE) * PIXEL_SIZE
        y = (y // PIXEL_SIZE) * PIXEL_SIZE

        if self.is_drawing_polygon:
            self.vertices.append((x, y))
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="blue", tags="vertex_marker")
            if len(self.vertices) > 1:
                x_prev, y_prev = self.vertices[-2]
                self._draw_line_on_canvas(x_prev, y_prev, x, y)
            if not self.finish_polygon_button['state'] == tk.NORMAL:
                 self.finish_polygon_button.config(state=tk.NORMAL)

        elif self.selected_algorithm.get() in ["seed_fill_simple", "seed_fill_scanline"] and not self.debug_generator:
            if not self.vertices or not self.boundary_pixels:
                messagebox.showerror("Error", "Please draw and finish a polygon first.")
                return
            self.seed_point = (x, y)
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", outline="red", tags="seed_marker")
            self.status_bar.config(text=f"Seed point selected at ({x},{y}). Click 'Fill Polygon'.")
            self.fill_button.config(state=tk.NORMAL) # Enable fill after seed is selected
        
        elif self.debug_mode.get() and self.debug_generator:
             self.status_bar.config(text="Debug mode active. Click 'Next Step'.")
        else:
            self.status_bar.config(text="Polygon finished. Select algorithm and click 'Fill Polygon' or 'Draw Polygon' for new.")


    def finish_polygon(self):
        if len(self.vertices) < 3:
            messagebox.showerror("Error", "A polygon needs at least 3 vertices.")
            return
        if self.vertices: # Close the polygon
            x_last, y_last = self.vertices[-1]
            x_first, y_first = self.vertices[0]
            self._draw_line_on_canvas(x_last, y_last, x_first, y_first)

        self.is_drawing_polygon = False
        self.finish_polygon_button.config(state=tk.DISABLED)
        self.draw_button.config(state=tk.NORMAL)
        
        if self.selected_algorithm.get() in ["seed_fill_simple", "seed_fill_scanline"]:
            self.status_bar.config(text="Polygon finished. Click inside to set seed point, then 'Fill Polygon'.")
        else:
            self.fill_button.config(state=tk.NORMAL)
            self.status_bar.config(text="Polygon finished. Click 'Fill Polygon'.")


    def start_fill(self):
        if not self.vertices or not self.boundary_pixels:
            messagebox.showerror("Error", "Please draw and finish a polygon first.")
            return

        algo = self.selected_algorithm.get()
        if algo in ["seed_fill_simple", "seed_fill_scanline"] and not self.seed_point:
            messagebox.showerror("Error", "Please select a seed point inside the polygon for this algorithm.")
            return

        # Clear previous fill and debug markers
        self.canvas.delete("filled_pixel")
        self.canvas.delete("debug_highlight")
        self.canvas.delete("seed_marker") # If any old one
        for item_id in self.filled_pixels_display.values():
            self.canvas.delete(item_id)
        self.filled_pixels_display.clear()

        self.debug_text_area.config(state=tk.NORMAL)
        self.debug_text_area.delete(1.0, tk.END)
        self.debug_text_area.config(state=tk.DISABLED)
        self.debug_step_count = 0

        fill_color = FILL_COLOR_SEED if "seed" in algo else FILL_COLOR_SCANLINE

        if self.debug_mode.get():
            self.next_step_button.config(state=tk.NORMAL)
            self.fill_button.config(state=tk.DISABLED) # Disable fill while debugging
            if algo == "scanline_ordered_edge_list":
                self.debug_generator = self._scanline_ordered_edge_list_debug(fill_color)
            elif algo == "scanline_ael":
                self.debug_generator = self._scanline_ael_debug(fill_color)
            elif algo == "seed_fill_simple":
                self.debug_generator = self._seed_fill_simple_debug(self.seed_point, fill_color)
            elif algo == "seed_fill_scanline":
                self.debug_generator = self._scanline_seed_fill_debug(self.seed_point, fill_color)
            self.execute_next_debug_step() # Execute the first step
        else:
            self.next_step_button.config(state=tk.DISABLED)
            if algo == "scanline_ordered_edge_list":
                for _ in self._scanline_ordered_edge_list_debug(fill_color): pass
            elif algo == "scanline_ael":
                for _ in self._scanline_ael_debug(fill_color): pass
            elif algo == "seed_fill_simple":
                for _ in self._seed_fill_simple_debug(self.seed_point, fill_color): pass
            elif algo == "seed_fill_scanline":
                for _ in self._scanline_seed_fill_debug(self.seed_point, fill_color): pass
            self.status_bar.config(text="Fill complete.")
            # self.fill_button.config(state=tk.DISABLED) # Disable after one fill

    def execute_next_debug_step(self):
        if self.debug_generator:
            try:
                next(self.debug_generator)
                self.debug_step_count += 1
                self.status_bar.config(text=f"Debug Step: {self.debug_step_count}. Click 'Next Step'.")
            except StopIteration:
                self.status_bar.config(text="Debug fill complete.")
                self.next_step_button.config(state=tk.DISABLED)
                # self.fill_button.config(state=tk.NORMAL if not self.selected_algorithm.get() in ["seed_fill_simple", "seed_fill_scanline"] else tk.DISABLED)
                self.debug_generator = None
            except Exception as e:
                messagebox.showerror("Debug Error", f"An error occurred during debug: {e}")
                self.reset_debug()

    def _append_debug_info(self, info):
        self.debug_text_area.config(state=tk.NORMAL)
        self.debug_text_area.insert(tk.END, info + "\n")
        self.debug_text_area.see(tk.END) # Scroll to the end
        self.debug_text_area.config(state=tk.DISABLED)

    def _highlight_scanline(self, y):
        self.canvas.delete("debug_scanline_highlight")
        self.canvas.create_line(0, y, CANVAS_WIDTH, y, fill="magenta", width=1, tags="debug_scanline_highlight", dash=(4, 2))

    def _highlight_points(self, points, color="orange", tag="debug_point_highlight", size=3):
        self.canvas.delete(tag)
        for x, y in points:
            self.canvas.create_oval(x - size, y - size, x + size, y + size, fill=color, outline=color, tags=tag)
            
    def reset_debug(self):
        self.debug_generator = None
        self.next_step_button.config(state=tk.DISABLED)
        self.canvas.delete("debug_highlight")
        self.canvas.delete("debug_scanline_highlight")
        self.canvas.delete("debug_point_highlight")
        self.canvas.delete("seed_marker")
        # Re-enable fill button based on current state
        if self.vertices and self.boundary_pixels:
            if self.selected_algorithm.get() in ["seed_fill_simple", "seed_fill_scanline"]:
                if self.seed_point:
                    self.fill_button.config(state=tk.NORMAL)
                else:
                    self.fill_button.config(state=tk.DISABLED)
                    self.status_bar.config(text="Select seed point for this algorithm.")
            else:
                self.fill_button.config(state=tk.NORMAL)
        else:
            self.fill_button.config(state=tk.DISABLED)


    def clear_all(self):
        self.canvas.delete("all")
        self.vertices = []
        self.current_polygon_edges = []
        self.boundary_pixels.clear()
        for item_id in self.filled_pixels_display.values():
            self.canvas.delete(item_id)
        self.filled_pixels_display.clear()
        self.seed_point = None
        self.is_drawing_polygon = True # Reset to drawing mode
        
        self.draw_button.config(state=tk.DISABLED)
        self.finish_polygon_button.config(state=tk.DISABLED)
        self.fill_button.config(state=tk.DISABLED)
        self.next_step_button.config(state=tk.DISABLED)
        
        self.debug_text_area.config(state=tk.NORMAL)
        self.debug_text_area.delete(1.0, tk.END)
        self.debug_text_area.config(state=tk.DISABLED)
        self.debug_generator = None
        self.status_bar.config(text="Mode: Drawing Polygon. Click to add vertices.")

    # --- Algorithm Implementations (as generators for debugging) ---

    def _scanline_ordered_edge_list_debug(self, fill_color):
        if not self.vertices or len(self.vertices) < 3:
            self._append_debug_info("Error: Polygon not defined.")
            return

        edges = []
        num_vertices = len(self.vertices)
        for i in range(num_vertices):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % num_vertices]
            # Ignore horizontal edges
            if p1[1] == p2[1]:
                continue
            # Ensure p1 is the upper vertex
            if p1[1] > p2[1]:
                p1, p2 = p2, p1
            
            slope_inv = float('inf') if p1[0] == p2[0] else (p2[0] - p1[0]) / (p2[1] - p1[1])
            edges.append({'y_min': p1[1], 'y_max': p2[1], 'x_at_y_min': p1[0], 'slope_inv': slope_inv})

        if not edges:
            self._append_debug_info("No non-horizontal edges found.")
            return

        min_scanline = min(e['y_min'] for e in edges)
        max_scanline = max(e['y_max'] for e in edges)
        
        self._append_debug_info(f"Polygon Vertices: {self.vertices}")
        self._append_debug_info(f"Edges for scanline processing (y_min, y_max, x_at_y_min, 1/m):")
        for i, e in enumerate(edges):
             self._append_debug_info(f"  Edge {i+1}: {e}")
        yield # Step 1: Show Edges

        all_intersections = []
        self._append_debug_info("\nStep 1: Finding all intersections with scanlines...")
        for y in range(min_scanline, max_scanline): # Scanlines are between pixels
            intersections_for_y = []
            for edge in edges:
                if y >= edge['y_min'] and y < edge['y_max']: # Edge crosses this scanline
                    if edge['slope_inv'] == float('inf'): # Vertical edge
                        x_intersect = edge['x_at_y_min']
                    else:
                        x_intersect = edge['x_at_y_min'] + (y - edge['y_min']) * edge['slope_inv']
                    intersections_for_y.append(round(x_intersect)) # Store as (y, x) for sorting later
            
            # Add (y, x) for sorting
            for x_val in intersections_for_y:
                all_intersections.append((y, x_val))

            if intersections_for_y: # Only log if there are intersections
                self._append_debug_info(f"  Scanline y={y}: Raw intersections X = {sorted(intersections_for_y)}")
        yield # Step 2: Show all raw intersections per scanline

        self._append_debug_info("\nStep 2: Sorting all intersections (by y, then x)...")
        all_intersections.sort() # Sort by y, then by x
        self._append_debug_info(f"  Sorted (y,x) list: {all_intersections}")
        yield # Step 3: Show sorted list

        self._append_debug_info("\nStep 3: Filling spans...")
        current_y = -1
        active_x_intersections = []

        for i in range(len(all_intersections)):
            y, x = all_intersections[i]

            if y != current_y: # New scanline
                # Process previous scanline's intersections if any
                if active_x_intersections:
                    self._highlight_scanline(current_y)
                    active_x_intersections.sort() # Sort x-intersections for the current scanline
                    self._append_debug_info(f"  Scanline y={current_y}: Sorted X = {active_x_intersections}")
                    self._highlight_points([(xi, current_y) for xi in active_x_intersections])
                    yield # Show sorted intersections for this scanline

                    for j in range(0, len(active_x_intersections) - 1, 2):
                        x_start = active_x_intersections[j]
                        x_end = active_x_intersections[j+1]
                        self._append_debug_info(f"    Fill span: [{x_start}, {x_end-1}] on y={current_y}")
                        for fill_x in range(x_start, x_end): # Fill up to x_end-1
                            self._draw_pixel(fill_x, current_y, fill_color, tag="filled_pixel")
                        yield # Show filled span
                    active_x_intersections = []
                current_y = y
            
            active_x_intersections.append(x)

        # Process the very last scanline's intersections
        if active_x_intersections:
            self._highlight_scanline(current_y)
            active_x_intersections.sort()
            self._append_debug_info(f"  Scanline y={current_y}: Sorted X = {active_x_intersections}")
            self._highlight_points([(xi, current_y) for xi in active_x_intersections])
            yield # Show sorted intersections for this scanline

            for j in range(0, len(active_x_intersections) - 1, 2):
                x_start = active_x_intersections[j]
                x_end = active_x_intersections[j+1]
                self._append_debug_info(f"    Fill span: [{x_start}, {x_end-1}] on y={current_y}")
                for fill_x in range(x_start, x_end):
                    self._draw_pixel(fill_x, current_y, fill_color, tag="filled_pixel")
                yield # Show filled span
        
        self.canvas.delete("debug_scanline_highlight")
        self.canvas.delete("debug_point_highlight")
        self._append_debug_info("Algorithm finished.")


    def _scanline_ael_debug(self, fill_color):
        if not self.vertices or len(self.vertices) < 3:
            self._append_debug_info("Error: Polygon not defined.")
            return

        # Build Edge Table (ET)
        # ET is a dictionary where keys are y_min scanlines
        # Values are lists of edges starting at that y_min
        # Edge: {'y_max': int, 'x_current': float, 'slope_inv': float}
        ET = {}
        min_scanline_poly = float('inf')
        max_scanline_poly = float('-inf')

        num_vertices = len(self.vertices)
        for i in range(num_vertices):
            p1_orig = self.vertices[i]
            p2_orig = self.vertices[(i + 1) % num_vertices]

            # Ignore horizontal edges
            if p1_orig[1] == p2_orig[1]:
                continue

            # Ensure p1 is the upper vertex (smaller y)
            p1 = list(p1_orig)
            p2 = list(p2_orig)
            if p1[1] > p2[1]:
                p1, p2 = p2, p1
            
            y_min = p1[1]
            y_max = p2[1] # y_max is exclusive for AEL processing (edge active up to y_max-1)
            x_at_y_min = float(p1[0])
            slope_inv = float('inf') if p1[0] == p2[0] else (p2[0] - p1[0]) / (p2[1] - p1[1])

            edge_data = {'y_max': y_max, 'x_current': x_at_y_min, 'slope_inv': slope_inv}
            
            if y_min not in ET:
                ET[y_min] = []
            ET[y_min].append(edge_data)

            min_scanline_poly = min(min_scanline_poly, y_min)
            max_scanline_poly = max(max_scanline_poly, y_max) # y_max here is the actual max y of edge

        if not ET:
            self._append_debug_info("No non-horizontal edges found.")
            return

        self._append_debug_info("Step 1: Edge Table (ET) Construction")
        for y_key in sorted(ET.keys()):
            self._append_debug_info(f"  ET[{y_key}]:")
            for edge in ET[y_key]:
                 self._append_debug_info(f"    y_max={edge['y_max']}, x_start={edge['x_current']:.2f}, 1/m={edge['slope_inv']:.2f}")
        yield # Show ET

        AEL = [] # Active Edge List
        self._append_debug_info("\nStep 2: Scanline Processing with AEL")
        for y in range(min_scanline_poly, max_scanline_poly): # Scan from y_min up to y_max-1 of polygon
            self._highlight_scanline(y)
            self._append_debug_info(f"\n--- Scanline y={y} ---")

            # 1. Remove edges from AEL whose y_max == y
            AEL = [edge for edge in AEL if edge['y_max'] > y]
            if AEL: self._append_debug_info(f"  AEL after removing expired edges (y_max <= {y}): {len(AEL)} edges")
            else: self._append_debug_info(f"  AEL empty after removing expired edges.")


            # 2. Add edges from ET[y] to AEL
            if y in ET:
                AEL.extend(ET[y])
                self._append_debug_info(f"  Added {len(ET[y])} edges from ET[{y}] to AEL. AEL now has {len(AEL)} edges.")
            
            if not AEL:
                yield # Show scanline highlight even if AEL is empty
                continue

            # 3. Sort AEL by x_current
            AEL.sort(key=lambda edge: edge['x_current'])
            self._append_debug_info(f"  AEL sorted by x_current ({len(AEL)} edges):")
            for i, edge in enumerate(AEL):
                self._append_debug_info(f"    AEL[{i}]: x={edge['x_current']:.2f}, y_max={edge['y_max']}, 1/m={edge['slope_inv']:.2f}")
            
            active_xs = [round(edge['x_current']) for edge in AEL]
            self._highlight_points([(x_val, y) for x_val in active_xs])
            yield # Show sorted AEL and highlighted intersection points

            # 4. Fill spans
            self._append_debug_info("  Filling spans:")
            for i in range(0, len(AEL) -1 , 2):
                x_start = round(AEL[i]['x_current'])
                x_end = round(AEL[i+1]['x_current'])
                self._append_debug_info(f"    Fill [{x_start}, {x_end-1}] on y={y}")
                for fill_x in range(x_start, x_end): # Fill up to x_end-1
                    self._draw_pixel(fill_x, y, fill_color, tag="filled_pixel")
            yield # Show filled spans for current scanline

            # 5. Update x_current for edges in AEL (for next scanline)
            for edge in AEL:
                if edge['slope_inv'] != float('inf'): # Not for vertical lines
                    edge['x_current'] += edge['slope_inv']
            self._append_debug_info("  Updated x_current in AEL for next scanline.")

        self.canvas.delete("debug_scanline_highlight")
        self.canvas.delete("debug_point_highlight")
        self._append_debug_info("\nAlgorithm finished.")


    def _is_pixel_valid_for_seed(self, x, y, fill_color):
        """Checks if pixel is within canvas, not boundary, and not already fill_color."""
        if not (0 <= x < CANVAS_WIDTH and 0 <= y < CANVAS_HEIGHT):
            return False
        if (x, y) in self.boundary_pixels:
            return False
        # Check if already filled (by checking its display item's color, or a separate set)
        # For simplicity, we use the self.filled_pixels_display and assume its color indicates fill
        if (x,y) in self.filled_pixels_display: # and self.canvas.itemcget(self.filled_pixels_display[(x,y)], "fill") == fill_color:
            # A more robust way: keep a set of (x,y) that are filled with target fill_color
            # For now, if it's in filled_pixels_display, assume it's filled (could be old highlight)
            # This simple check is fine IF we clear filled_pixels_display before each fill run.
            # A better check:
            #   item_id = self.filled_pixels_display.get((x,y))
            #   if item_id and self.canvas.itemcget(item_id, "fill") == fill_color: return False
            # Current implementation _draw_pixel replaces, so this check isn't perfect if colors vary.
            # Let's add a dedicated set for filled pixels.
            if (x,y) in self.already_filled_with_target_color_set: # Requires this set to be managed
                return False
        return True

    def _seed_fill_simple_debug(self, seed_pt, fill_color):
        if not seed_pt:
            self._append_debug_info("Error: Seed point not set.")
            return
        
        sx, sy = seed_pt
        # Initialize a set to keep track of pixels filled with *this specific* fill_color
        # This is important because self.filled_pixels_display might contain other things
        self.already_filled_with_target_color_set = set() 

        if (sx, sy) in self.boundary_pixels:
            self._append_debug_info(f"Seed point ({sx},{sy}) is on the boundary. Cannot fill.")
            return
        if not (0 <= sx < CANVAS_WIDTH and 0 <= sy < CANVAS_HEIGHT):
            self._append_debug_info(f"Seed point ({sx},{sy}) is outside canvas.")
            return

        stack = [(sx, sy)]
        self._append_debug_info(f"Initial Seed: ({sx},{sy}). Stack: {stack}")
        self._highlight_points([(sx,sy)], color="purple", tag="seed_highlight", size=4)
        yield

        while stack:
            self.canvas.delete("current_pixel_highlight")
            px, py = stack.pop()
            self._append_debug_info(f"\nPopped ({px},{py}). Stack: {stack}")
            self._highlight_points([(px,py)], color=HIGHLIGHT_COLOR, tag="current_pixel_highlight", size=2)
            yield

            if self._is_pixel_valid_for_seed(px, py, fill_color):
                self._draw_pixel(px, py, fill_color, tag="filled_pixel")
                self.already_filled_with_target_color_set.add((px,py))
                self._append_debug_info(f"  Filled ({px},{py}).")
                yield # Show filled pixel

                # 4-connectivity neighbors
                neighbors = [(px, py - PIXEL_SIZE), (px, py + PIXEL_SIZE),
                             (px - PIXEL_SIZE, py), (px + PIXEL_SIZE, py)]
                
                pushed_neighbors = []
                for nx, ny in neighbors:
                    if self._is_pixel_valid_for_seed(nx, ny, fill_color) and (nx,ny) not in stack : # Check not in stack already
                        stack.append((nx, ny))
                        pushed_neighbors.append((nx,ny))
                if pushed_neighbors:
                    self._append_debug_info(f"  Pushed valid neighbors to stack: {pushed_neighbors}")
                    self._append_debug_info(f"  Stack now: {stack}")
                else:
                    self._append_debug_info(f"  No new valid neighbors to push.")
                self._highlight_points(stack, color="blue", tag="stack_viz_highlight", size=1)
                yield # Show stack and neighbors considered
            else:
                 self._append_debug_info(f"  Pixel ({px},{py}) is invalid (boundary, outside, or already filled). Skipped.")
                 yield

        self.canvas.delete("current_pixel_highlight")
        self.canvas.delete("stack_viz_highlight")
        self.canvas.delete("seed_highlight")
        self._append_debug_info("\nAlgorithm finished.")
        del self.already_filled_with_target_color_set # Clean up helper set

    def _scanline_seed_fill_debug(self, seed_pt, fill_color):
        if not seed_pt:
            self._append_debug_info("Error: Seed point not set.")
            return

        sx, sy = seed_pt
        self.already_filled_with_target_color_set = set()

        if (sx, sy) in self.boundary_pixels:
            self._append_debug_info(f"Seed point ({sx},{sy}) is on the boundary. Cannot fill.")
            return
        if not (0 <= sx < CANVAS_WIDTH and 0 <= sy < CANVAS_HEIGHT):
            self._append_debug_info(f"Seed point ({sx},{sy}) is outside canvas.")
            return

        stack = [(sx, sy)] # Stack stores one seed pixel per horizontal segment to fill
        self._append_debug_info(f"Initial Seed: ({sx},{sy}). Stack: {stack}")
        self._highlight_points([(sx,sy)], color="purple", tag="seed_highlight", size=4)

        yield

        while stack:
            self.canvas.delete("current_span_highlight")
            self.canvas.delete("new_seeds_highlight")
            
            curr_x, curr_y = stack.pop()
            self._append_debug_info(f"\nPopped seed ({curr_x},{curr_y}) for scanline y={curr_y}. Stack: {stack}")
            yield

            # Find span to the left
            x_left = curr_x
            while self._is_pixel_valid_for_seed(x_left - PIXEL_SIZE, curr_y, fill_color):
                x_left -= PIXEL_SIZE
            
            # Find span to the right
            x_right = curr_x
            while self._is_pixel_valid_for_seed(x_right + PIXEL_SIZE, curr_y, fill_color):
                x_right += PIXEL_SIZE
            
            self._append_debug_info(f"  Identified span on y={curr_y}: [{x_left}, {x_right}]")
            self.canvas.create_line(x_left, curr_y + PIXEL_SIZE/2, x_right + PIXEL_SIZE, curr_y + PIXEL_SIZE/2, 
                                    fill="orange", width=3, tags="current_span_highlight")
            yield # Show identified span

            # Fill the span
            self._append_debug_info(f"  Filling span [{x_left},{x_right}] on y={curr_y}")
            for x_fill in range(x_left, x_right + PIXEL_SIZE, PIXEL_SIZE):
                if self._is_pixel_valid_for_seed(x_fill, curr_y, fill_color): # Double check before drawing
                    self._draw_pixel(x_fill, curr_y, fill_color, tag="filled_pixel")
                    self.already_filled_with_target_color_set.add((x_fill,curr_y))
            yield # Show filled span

            # Check scanline above and below for new seeds
            new_seeds_pushed = []
            for y_offset in [-PIXEL_SIZE, PIXEL_SIZE]: # Above and Below
                scan_y = curr_y + y_offset
                if not (0 <= scan_y < CANVAS_HEIGHT): continue

                self._append_debug_info(f"  Checking scanline y={scan_y} in range x=[{x_left},{x_right}] for new seeds...")
                
                x_scan = x_left
                while x_scan <= x_right:
                    # Find a seed: skip non-valid pixels
                    while x_scan <= x_right and not self._is_pixel_valid_for_seed(x_scan, scan_y, fill_color):
                        x_scan += PIXEL_SIZE
                    
                    if x_scan <= x_right: # Found a potential start of a new span
                        # Now scan right to find the end of this potential new span
                        seed_candidate_x = x_scan
                        while x_scan <= x_right and self._is_pixel_valid_for_seed(x_scan, scan_y, fill_color):
                             # The text says: " крайний правый пиксел отмечается как затравочный"
                             # So the actual seed should be the rightmost of this segment.
                            seed_candidate_x = x_scan 
                            x_scan += PIXEL_SIZE
                        
                        # Push the rightmost pixel of the valid segment as a seed
                        if (seed_candidate_x, scan_y) not in stack:
                            stack.append((seed_candidate_x, scan_y))
                            new_seeds_pushed.append((seed_candidate_x, scan_y))
                    x_scan += PIXEL_SIZE # Ensure progress if no valid pixel found or after a segment
            
            if new_seeds_pushed:
                self._append_debug_info(f"  New seeds pushed to stack: {new_seeds_pushed}")
                self._append_debug_info(f"  Stack now: {stack}")
                self._highlight_points(new_seeds_pushed, color="green", tag="new_seeds_highlight", size=2)
            else:
                self._append_debug_info(f"  No new seeds found on adjacent scanlines.")
            yield # Show new seeds found

        self.canvas.delete("current_span_highlight")
        self.canvas.delete("new_seeds_highlight")
        self.canvas.delete("seed_highlight")
        self._append_debug_info("\nAlgorithm finished.")
        del self.already_filled_with_target_color_set # Clean up


if __name__ == '__main__':
    root = tk.Tk()
    app = PolygonFillerApp(root)
    root.mainloop()