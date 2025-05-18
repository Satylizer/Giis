import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend for GUI elements
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon as MplPolygon # Renamed to avoid conflict
from matplotlib.patches import Circle

class InteractiveTriangulationVoronoi:
    def __init__(self):
        self.points = []
        self.fig, (self.ax_delaunay, self.ax_voronoi) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.canvas.manager.set_window_title('Delaunay Triangulation & Voronoi Diagram Lab')

        self.show_voronoi_vertices = True
        self.show_circumcircles = False # Off by default for less clutter

        self.setup_plots()
        self.connect_event_handlers()
        self.add_buttons()
        self.update_plots() # Initial empty plot

        plt.show()

    def setup_plots(self):
        # Delaunay Plot Setup
        self.ax_delaunay.set_title('Delaunay Triangulation (Click to add points)')
        self.ax_delaunay.set_xlabel('X')
        self.ax_delaunay.set_ylabel('Y')
        self.ax_delaunay.set_aspect('equal', adjustable='box')
        self.ax_delaunay.grid(True)

        # Voronoi Plot Setup
        self.ax_voronoi.set_title('Voronoi Diagram')
        self.ax_voronoi.set_xlabel('X')
        self.ax_voronoi.set_ylabel('Y')
        # self.ax_voronoi.set_aspect('equal', adjustable='box') # Can make it too large
        self.ax_voronoi.grid(True)

        self.fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for buttons

    def connect_event_handlers(self):
        # Connect click event to the Delaunay plot for adding points
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def add_buttons(self):
        ax_clear = plt.axes([0.7, 0.01, 0.1, 0.04]) # x, y, width, height
        self.btn_clear = Button(ax_clear, 'Clear Points')
        self.btn_clear.on_clicked(self.clear_points)

        ax_toggle_vv = plt.axes([0.55, 0.01, 0.13, 0.04])
        self.btn_toggle_vv = Button(ax_toggle_vv, 'Toggle Voronoi Vertices')
        self.btn_toggle_vv.on_clicked(self.toggle_voronoi_vertices)

        ax_toggle_cc = plt.axes([0.35, 0.01, 0.18, 0.04])
        self.btn_toggle_cc = Button(ax_toggle_cc, 'Toggle Circumcircles')
        self.btn_toggle_cc.on_clicked(self.toggle_circumcircles)


    def on_click(self, event):
        # Add point only if clicked within the Delaunay axes
        if event.inaxes == self.ax_delaunay:
            if event.button == 1: # Left mouse button
                self.points.append([event.xdata, event.ydata])
                self.update_plots()

    def clear_points(self, event):
        self.points = []
        self.update_plots()

    def toggle_voronoi_vertices(self, event):
        self.show_voronoi_vertices = not self.show_voronoi_vertices
        self.update_plots() # Re-plot Voronoi

    def toggle_circumcircles(self, event):
        self.show_circumcircles = not self.show_circumcircles
        self.update_plots() # Re-plot Delaunay

    def update_plots(self):
        # Clear previous plots
        self.ax_delaunay.clear()
        self.ax_voronoi.clear()
        self.setup_plots() # Re-apply titles, labels, grid, aspect ratio

        points_np = np.array(self.points)

        # Plot current points on both axes
        if len(points_np) > 0:
            self.ax_delaunay.plot(points_np[:, 0], points_np[:, 1], 'o', color='red', markersize=5)
            self.ax_voronoi.plot(points_np[:, 0], points_np[:, 1], 'o', color='red', markersize=5, label="Sites")

        # Delaunay Triangulation
        if len(points_np) >= 3:
            try:
                delaunay_tri = Delaunay(points_np)
                self.ax_delaunay.triplot(points_np[:, 0], points_np[:, 1], delaunay_tri.simplices, color='blue')

                if self.show_circumcircles:
                    for simplex in delaunay_tri.simplices:
                        triangle_pts = points_np[simplex]
                        p1, p2, p3 = triangle_pts[0], triangle_pts[1], triangle_pts[2]
                        # Check for collinearity before D calculation
                        # A more robust collinearity check:
                        # If (y2-y1)*(x3-x2) == (y3-y2)*(x2-x1), then collinear
                        # Using area is also good
                        area_check = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

                        if area_check > 1e-9: # Not collinear
                            D_val = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                            if abs(D_val) > 1e-9: # Avoid division by zero
                                center_x = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + \
                                            (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + \
                                            (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / D_val
                                center_y = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + \
                                            (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + \
                                            (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / D_val
                                radius = np.sqrt((p1[0] - center_x)**2 + (p1[1] - center_y)**2)
                                circum = Circle((center_x, center_y), radius, color='purple', fill=False, linestyle='--', alpha=0.5)
                                self.ax_delaunay.add_patch(circum)
            except Exception as e:
                print(f"Delaunay error: {e}") # e.g., QhullError if points are collinear

        # Voronoi Diagram
        if len(points_np) >= 2: # Voronoi needs at least 2 points for a diagram
            try:
                vor = Voronoi(points_np)
                voronoi_plot_2d(vor, ax=self.ax_voronoi, show_points=False, # Points already plotted
                                show_vertices=self.show_voronoi_vertices,
                                line_colors='orange', line_width=1.5, line_alpha=0.7, point_size=3)
            except Exception as e:
                 print(f"Voronoi error: {e}") # e.g., QhullError

        # Set plot limits dynamically for Voronoi, or keep them fixed
        if len(points_np) > 0:
            min_x, max_x = np.min(points_np[:,0]), np.max(points_np[:,0])
            min_y, max_y = np.min(points_np[:,1]), np.max(points_np[:,1])
            padding_x = max(1.0, (max_x - min_x) * 0.2) # Ensure some padding
            padding_y = max(1.0, (max_y - min_y) * 0.2)

            self.ax_delaunay.set_xlim(min_x - padding_x, max_x + padding_x)
            self.ax_delaunay.set_ylim(min_y - padding_y, max_y + padding_y)
            self.ax_voronoi.set_xlim(min_x - padding_x, max_x + padding_x)
            self.ax_voronoi.set_ylim(min_y - padding_y, max_y + padding_y)
        else: # Default limits for empty plot
            self.ax_delaunay.set_xlim(0, 10)
            self.ax_delaunay.set_ylim(0, 10)
            self.ax_voronoi.set_xlim(0, 10)
            self.ax_voronoi.set_ylim(0, 10)

        self.ax_voronoi.legend(loc='upper right')
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    # --- Instructions ---
    # 1. Save this code as a Python file (e.g., lab_program.py).
    # 2. Make sure you have the necessary libraries installed:
    #    pip install numpy matplotlib scipy
    #    (If you still get Qt errors after `matplotlib.use('TkAgg')`,
    #     you might need to ensure Tkinter is properly set up with your Python,
    #     or try `pip install PyQt5` as a fallback for Matplotlib's backend needs.)
    # 3. Open a terminal or command prompt.
    # 4. Activate your Python virtual environment if you are using one.
    # 5. Navigate to the directory where you saved the file.
    # 6. Run the script: python lab_program.py

    app = InteractiveTriangulationVoronoi()