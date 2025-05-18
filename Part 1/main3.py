import tkinter as tk
from tkinter import ttk
import numpy as np
import math
from math import sqrt, factorial

class GraphicsEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Графический редактор")
        
        # Настройка размеров
        self.canvas_width = 800
        self.canvas_height = 600
        self.grid_step = 20
        
        # Настройка переменных
        self.mode = tk.StringVar(value="lines")  # Режим работы: lines/parametric/conic
        self.curve_type = tk.StringVar(value="Bezier")  # Для параметрических кривых
        self.line_algorithm = tk.StringVar(value="DDA")
        self.conic_type = tk.StringVar(value="Circle")
        self.debug_mode = tk.BooleanVar(value=False)
        
        # Переменные для построения
        self.control_points = []
        self.current_point = None
        self.start_point = None
        self.preview_shape = None
        self.steps = []
        self.current_step = 0
        self.edit_mode = tk.BooleanVar(value=False)
        self.snap_mode = tk.BooleanVar(value=True)
        
        # Матрицы для кривых
        self.hermite_matrix = np.array([
            [2, -2, 1, 1],
            [-3, 3, -2, -1],
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ])
        
        self.bezier_matrix = np.array([
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 3, 0, 0],
            [1, 0, 0, 0]
        ])
        
        # Окна и холсты
        self.debug_window = None
        self.debug_canvas = None
        self.grid_lines = []
        
        # Создание интерфейса
        self.create_menu()
        self.create_toolbars()
        self.create_canvas()
        
        # Привязка событий
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_motion)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # Меню Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Новый", command=self.new_canvas)
        file_menu.add_command(label="Очистить", command=self.clear_canvas)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)
        
        # Меню Кривые
        curve_menu = tk.Menu(menubar, tearoff=0)
        curve_menu.add_radiobutton(label="Эрмита", variable=self.curve_type, value="Hermite")
        curve_menu.add_radiobutton(label="Безье", variable=self.curve_type, value="Bezier")
        curve_menu.add_radiobutton(label="B-сплайн", variable=self.curve_type, value="B-spline")
        menubar.add_cascade(label="Кривые", menu=curve_menu)
        
        # Меню для отрезков
        lines_menu = tk.Menu(menubar, tearoff=0)
        lines_menu.add_radiobutton(label="ЦДА", variable=self.line_algorithm, value="DDA")
        lines_menu.add_radiobutton(label="Брезенхем", variable=self.line_algorithm, value="Bresenham")
        lines_menu.add_radiobutton(label="Ву", variable=self.line_algorithm, value="Wu")
        menubar.add_cascade(label="Отрезки", menu=lines_menu)
        
        # Меню для конических сечений
        conic_menu = tk.Menu(menubar, tearoff=0)
        conic_menu.add_radiobutton(label="Окружность", variable=self.conic_type, value="Circle")
        conic_menu.add_radiobutton(label="Эллипс", variable=self.conic_type, value="Ellipse")
        conic_menu.add_radiobutton(label="Гипербола", variable=self.conic_type, value="Hyperbola")
        conic_menu.add_radiobutton(label="Парабола", variable=self.conic_type, value="Parabola")
        menubar.add_cascade(label="Конические сечения", menu=conic_menu)
        
        # Меню Настройки
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_checkbutton(label="Привязка к сетке", variable=self.snap_mode)
        settings_menu.add_checkbutton(label="Режим отладки", variable=self.debug_mode, command=self.toggle_debug)
        settings_menu.add_checkbutton(label="Режим редактирования", variable=self.edit_mode, command=self.toggle_edit_mode)
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        
        self.root.config(menu=menubar)

    def create_toolbars(self):
        # Панель для выбора режима
        mode_toolbar = ttk.Frame(self.root)
        ttk.Label(mode_toolbar, text="Режим:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_toolbar, text="Кривые", variable=self.mode, value="parametric").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_toolbar, text="Отрезки", variable=self.mode, value="lines").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_toolbar, text="Конические", variable=self.mode, value="conic").pack(side=tk.LEFT)
        mode_toolbar.pack(side=tk.TOP, fill=tk.X)

    def create_canvas(self):
        self.canvas = tk.Canvas(self.root, 
                              width=self.canvas_width, 
                              height=self.canvas_height, 
                              bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    def snap_to_grid(self, x, y):
        if self.snap_mode.get():
            x = round(x / self.grid_step) * self.grid_step
            y = round(y / self.grid_step) * self.grid_step
        return x, y
    
    def on_click(self, event):
        x, y = self.snap_to_grid(event.x, event.y)
        
        if self.mode.get() == "parametric":
            if self.edit_mode.get():
                # Проверяем, кликнули ли на контрольную точку
                for i, point in enumerate(self.control_points):
                    if sqrt((point[0]-x)**2 + (point[1]-y)**2) < 10:
                        self.current_point = i
                        return
                self.current_point = None
            else:
                # Добавляем новую контрольную точку
                self.control_points.append((x, y))
                self.draw_curve()
        else:
            # Запоминаем начальную точку для фигур/отрезков
            self.start_point = (x, y)
    
    def on_drag(self, event):
        if self.edit_mode.get() and self.current_point is not None and self.mode.get() == "parametric":
            x, y = self.snap_to_grid(event.x, event.y)
            self.control_points[self.current_point] = (x, y)
            self.draw_curve()
        elif self.start_point and self.mode.get() != "parametric":
            self.preview_shape_event(event)

    def on_release(self, event):
        if self.mode.get() == "lines" and self.start_point:
            self.draw_line_event(event)
        elif self.mode.get() == "conic" and self.start_point:
            self.draw_conic_event(event)
        self.start_point = None
        self.preview_shape = None

    def on_motion(self, event):
        if self.edit_mode.get() and self.mode.get() == "parametric":
            x, y = event.x, event.y
            for i, point in enumerate(self.control_points):
                if sqrt((point[0]-x)**2 + (point[1]-y)**2) < 10:
                    self.canvas.config(cursor="hand2")
                    return
            self.canvas.config(cursor="")
        elif self.start_point and self.mode.get() != "parametric":
            self.preview_shape_event(event)

    def draw_curve(self):
        self.canvas.delete("all")
        
        # Рисуем контрольные точки
        for i, (x, y) in enumerate(self.control_points):
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="red")
            self.canvas.create_text(x, y-10, text=str(i+1))
        
        if len(self.control_points) < 2:
            return
            
        # Рисуем контрольный полигон
        points = []
        for x, y in self.control_points:
            points.extend([x, y])
        self.canvas.create_line(*points, fill="gray", dash=(2,2))
        
        # Рисуем кривую в зависимости от типа
        if self.curve_type.get() == "Hermite" and len(self.control_points) >= 4:
            self.draw_hermite_curve()
        elif self.curve_type.get() == "Bezier":
            self.draw_bezier_curve()
        elif self.curve_type.get() == "B-spline" and len(self.control_points) >= 4:
            self.draw_bspline_curve()
        
        # Подсвечиваем точки в режиме редактирования
        if self.edit_mode.get():
            self.highlight_control_points()

    def draw_hermite_curve(self):
        segments = len(self.control_points) // 4
        for i in range(segments):
            start_idx = i * 4
            if start_idx + 3 >= len(self.control_points):
                break
                
            P1 = self.control_points[start_idx]
            P4 = self.control_points[start_idx + 1]
            R1 = (self.control_points[start_idx + 2][0] - P1[0], 
                  self.control_points[start_idx + 2][1] - P1[1])
            R4 = (self.control_points[start_idx + 3][0] - P4[0], 
                  self.control_points[start_idx + 3][1] - P4[1])
            
            # Создаем матрицу геометрии
            G_x = np.array([[P1[0]], [P4[0]], [R1[0]], [R4[0]]])
            G_y = np.array([[P1[1]], [P4[1]], [R1[1]], [R4[1]]])
            
            # Умножаем матрицу Эрмита на матрицы геометрии
            C_x = np.dot(self.hermite_matrix, G_x)
            C_y = np.dot(self.hermite_matrix, G_y)
            
            points = []
            for t in np.linspace(0, 1, 50):
                T = np.array([t**3, t**2, t, 1])
                x = np.dot(T, C_x)[0]
                y = np.dot(T, C_y)[0]
                points.append((x, y))
            
            for j in range(len(points)-1):
                self.canvas.create_line(points[j][0], points[j][1], points[j+1][0], points[j+1][1], fill="blue", width=2)

    def draw_bezier_curve(self):
        n = len(self.control_points) - 1
        points = []
        
        for t in np.linspace(0, 1, 100):
            x, y = 0, 0
            for i in range(n + 1):
                coeff = self.combination(n, i) * (t**i) * ((1 - t)**(n - i))
                x += coeff * self.control_points[i][0]
                y += coeff * self.control_points[i][1]
            points.append((x, y))
        
        for i in range(len(points)-1):
            self.canvas.create_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1], fill="green", width=2)

    def combination(self, n, k):
        return factorial(n) / (factorial(k) * factorial(n - k))

    def draw_bspline_curve(self):
        n = len(self.control_points)
        if n < 4:
            return
            
        k = 3  # Степень кривой (кубический сплайн)
        knots = list(range(n + k + 1))  # Узловой вектор
        
        points = []
        for u in np.linspace(k, n, 100):
            x, y = 0, 0
            for i in range(n):
                basis = self.basis_function(i, k, u, knots)
                x += basis * self.control_points[i][0]
                y += basis * self.control_points[i][1]
            points.append((x, y))
        
        for i in range(len(points)-1):
            self.canvas.create_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1], fill="purple", width=2)

    def basis_function(self, i, k, u, knots):
        if k == 0:
            return 1 if knots[i] <= u < knots[i+1] else 0
        else:
            den1 = knots[i+k] - knots[i]
            term1 = 0 if den1 == 0 else ((u - knots[i])/den1) * self.basis_function(i, k-1, u, knots)
            
            den2 = knots[i+k+1] - knots[i+1]
            term2 = 0 if den2 == 0 else ((knots[i+k+1] - u)/den2) * self.basis_function(i+1, k-1, u, knots)
            
            return term1 + term2

    def highlight_control_points(self):
        for x, y in self.control_points:
            self.canvas.create_oval(x-6, y-6, x+6, y+6, outline="blue", width=2)

    def toggle_edit_mode(self):
        if self.mode.get() == "parametric":
            self.draw_curve()

    def preview_shape_event(self, event):
        if self.preview_shape:
            self.canvas.delete(self.preview_shape)
            
        x, y = self.snap_to_grid(event.x, event.y)
        x0, y0 = self.start_point
        
        if self.mode.get() == "lines":
            self.preview_shape = self.canvas.create_line(x0, y0, x, y, dash=(2,2))
        elif self.mode.get() == "conic":
            if self.conic_type.get() == "Circle":
                radius = int(sqrt((x-x0)**2 + (y-y0)**2))
                self.preview_shape = self.canvas.create_oval(
                    x0-radius, y0-radius,
                    x0+radius, y0+radius, dash=(2,2))
            elif self.conic_type.get() == "Ellipse":
                a = abs(x - x0)
                b = abs(y - y0)
                self.preview_shape = self.canvas.create_oval(
                    x0-a, y0-b,
                    x0+a, y0+b, dash=(2,2))

    def draw_line_event(self, event):
        x1, y1 = self.snap_to_grid(event.x, event.y)
        x0, y0 = self.start_point
        
        if self.line_algorithm.get() == "DDA":
            points = self.dda_algorithm(x0, y0, x1, y1)
        elif self.line_algorithm.get() == "Bresenham":
            points = self.bresenham_algorithm(x0, y0, x1, y1)
        else:
            points = self.wu_algorithm(x0, y0, x1, y1)
        
        self.steps = points
        self.draw_points(points)
        
        if self.debug_mode.get():
            self.show_debug_info()

    def dda_algorithm(self, x0, y0, x1, y1):
        points = []
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            return [(x0, y0)]
            
        x_inc = dx / steps
        y_inc = dy / steps
        
        x, y = x0, y0
        for _ in range(steps + 1):
            points.append((round(x), round(y)))
            x += x_inc
            y += y_inc
        return points

    def bresenham_algorithm(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steep = dy > dx
        
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            dx, dy = dy, dx
            
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            
        error = 0
        derr = dy
        y = y0
        ystep = 1 if y1 > y0 else -1
        
        for x in range(x0, x1 + 1):
            coord = (y, x) if steep else (x, y)
            points.append(coord)
            error += derr
            if 2 * error >= dx:
                y += ystep
                error -= dx
        return points

    def wu_algorithm(self, x0, y0, x1, y1):
        points = []
        def ipart(x): return int(x)
        def fpart(x): return x - int(x)
        def rfpart(x): return 1 - fpart(x)
        
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            
        dx = x1 - x0
        dy = y1 - y0
        gradient = dy / dx if dx != 0 else 1
        
        xend = round(x0)
        yend = y0 + gradient * (xend - x0)
        xgap = rfpart(x0 + 0.5)
        xpxl1 = xend
        ypxl1 = ipart(yend)
        if steep:
            points.extend([(ypxl1, xpxl1), (ypxl1+1, xpxl1)])
        else:
            points.extend([(xpxl1, ypxl1), (xpxl1, ypxl1+1)])
            
        intery = yend + gradient
        
        xend = round(x1)
        yend = y1 + gradient * (xend - x1)
        xgap = fpart(x1 + 0.5)
        xpxl2 = xend
        ypxl2 = ipart(yend)
        if steep:
            points.extend([(ypxl2, xpxl2), (ypxl2+1, xpxl2)])
        else:
            points.extend([(xpxl2, ypxl2), (xpxl2, ypxl2+1)])
            
        for x in range(xpxl1 + 1, xpxl2):
            if steep:
                points.append((ipart(intery), x))
                points.append((ipart(intery)+1, x))
            else:
                points.append((x, ipart(intery)))
                points.append((x, ipart(intery)+1))
            intery += gradient
        return points

    def draw_conic_event(self, event):
        x1, y1 = self.snap_to_grid(event.x, event.y)
        x0, y0 = self.start_point
        
        if self.conic_type.get() == "Circle":
            radius = int(sqrt((x1-x0)**2 + (y1-y0)**2))
            points = self.circle_bresenham(x0, y0, radius)
        elif self.conic_type.get() == "Ellipse":
            a = abs(x1 - x0)
            b = abs(y1 - y0)
            points = self.ellipse_midpoint(x0, y0, a, b)
        elif self.conic_type.get() == "Hyperbola":
            a = abs(x1 - x0) // 2
            b = abs(y1 - y0) // 2
            points = self.hyperbola(x0, y0, a, b)
        elif self.conic_type.get() == "Parabola":
            p = abs(x1 - x0)
            points = self.parabola(x0, y0, p)
        
        self.steps = points
        self.draw_points(points)
        
        if self.debug_mode.get():
            self.show_debug_info()

    def circle_bresenham(self, xc, yc, r):
        points = []
        x = 0
        y = r
        d = 3 - 2 * r
        while y >= x:
            points.extend([
                (xc+x, yc+y), (xc-x, yc+y),
                (xc+x, yc-y), (xc-x, yc-y),
                (xc+y, yc+x), (xc-y, yc+x),
                (xc+y, yc-x), (xc-y, yc-x)
            ])
            x += 1
            if d > 0:
                y -= 1
                d += 4 * (x - y) + 10
            else:
                d += 4 * x + 6
        return points

    def ellipse_midpoint(self, xc, yc, a, b):
        points = []
        x = 0
        y = b
        a_sq = a*a
        b_sq = b*b
        d = 4 * b_sq * ((x + 1)**2) + a_sq * ((2*y - 1)**2) - 4 * a_sq * b_sq
        
        while a_sq*(2*y - 1) > 2*b_sq*(x + 1):
            points.extend([
                (xc+x, yc+y), (xc-x, yc+y),
                (xc+x, yc-y), (xc-x, yc-y)
            ])
            if d < 0:
                d += 4*b_sq*(2*x + 3)
            else:
                d += 4*b_sq*(2*x + 3) - 8*a_sq*(y - 1)
                y -= 1
            x += 1
        
        d = b_sq*((2*x + 1)**2) + 4*a_sq*((y + 1)**2) - 4*a_sq*b_sq
        while y >= 0:
            points.extend([
                (xc+x, yc+y), (xc-x, yc+y),
                (xc+x, yc-y), (xc-x, yc-y)
            ])
            if d < 0:
                d += 4*a_sq*(2*y + 3)
            else:
                d += 4*a_sq*(2*y + 3) - 8*b_sq*(x + 1)
                x += 1
            y -= 1
        return points

    def hyperbola(self, x0, y0, a, b):
        points = []
        step = 1
        max_x = self.canvas_width // 2
        
        # Правая ветвь
        x = a
        while x <= max_x:
            y = b * math.sqrt((x/a)**2 - 1)
            points.append((x0 + x, y0 + int(y)))
            points.append((x0 + x, y0 - int(y)))
            x += step
        
        # Левая ветвь
        x = -a
        while x >= -max_x:
            y = b * math.sqrt((x/a)**2 - 1)
            points.append((x0 + x, y0 + int(y)))
            points.append((x0 + x, y0 - int(y)))
            x -= step
            
        return points

    def parabola(self, x0, y0, p):
        points = []
        step = 1
        max_x = self.canvas_width // 2
        
        # Правая ветвь
        x = 0
        while x <= max_x:
            y = (x**2) / (4*p)
            points.append((x0 + x, y0 + int(y)))
            points.append((x0 + x, y0 - int(y)))
            x += step
            
        # Левая ветвь
        x = 0
        while x >= -max_x:
            y = (x**2) / (4*p)
            points.append((x0 + x, y0 + int(y)))
            points.append((x0 + x, y0 - int(y)))
            x -= step
            
        return points

    def draw_points(self, points):
        for x, y in points:
            self.canvas.create_rectangle(x, y, x+1, y+1, fill="black")

    def new_canvas(self):
        self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.control_points = []
        self.start_point = None
        self.preview_shape = None
        self.steps = []
        self.current_step = 0

    def create_debug_window(self):
        if self.debug_window and self.debug_window.winfo_exists():
            return
        
        self.debug_window = tk.Toplevel(self.root)
        self.debug_window.title("Режим отладки")
        self.debug_window.geometry(f"{self.canvas_width}x{self.canvas_height}")
        self.debug_window.protocol("WM_DELETE_WINDOW", self.toggle_debug)
        self.debug_canvas = tk.Canvas(self.debug_window, 
                                    width=self.canvas_width, 
                                    height=self.canvas_height, 
                                    bg="white")
        self.debug_canvas.pack()
        self.draw_grid()

    def draw_grid(self):
        if not self.debug_canvas:
            return
        
        # Очистка предыдущей сетки
        for line in self.grid_lines:
            self.debug_canvas.delete(line)
        self.grid_lines = []
        
        # Рисование новой сетки
        for i in range(0, self.canvas_width, self.grid_step):
            self.grid_lines.append(self.debug_canvas.create_line(
                i, 0, i, self.canvas_height, fill="lightgray"))
        for i in range(0, self.canvas_height, self.grid_step):
            self.grid_lines.append(self.debug_canvas.create_line(
                0, i, self.canvas_width, i, fill="lightgray"))

    def toggle_debug(self):
        if self.debug_mode.get():
            self.create_debug_window()
            self.show_debug_info()
        else:
            if self.debug_window:
                self.debug_window.destroy()
            self.debug_window = None
            self.debug_canvas = None

    def show_debug_info(self):
        if not (self.debug_window and self.debug_canvas and self.steps):
            return
        
        self.debug_canvas.delete("all")
        self.draw_grid()
        
        scale = self.grid_step
        scaled_points = [(x//scale, y//scale) for x, y in self.steps]
        self.steps = scaled_points
        self.current_step = 0
        
        self.animate_step()

    def animate_step(self):
        if (self.current_step < len(self.steps) and 
            self.debug_canvas and 
            self.debug_window and 
            self.debug_window.winfo_exists()):
            
            scale = self.grid_step
            x, y = self.steps[self.current_step]
            
            # Рисуем точку с анимацией
            self.debug_canvas.create_rectangle(
                x*scale + 3, y*scale + 3,
                x*scale + scale - 3, y*scale + scale - 3,
                fill="red", tags="points", outline=""
            )
            
            self.current_step += 1
            self.root.after(50, self.animate_step)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphicsEditor(root)
    root.mainloop()