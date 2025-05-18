# -*- coding: utf-8 -*-
import pygame
import numpy as np
import math

SCREEN_WIDTH, SCREEN_HEIGHT = 900, 700
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)

INFO_TEXT_COLOR = GREEN
COORDS_TEXT_COLOR = YELLOW
INDEX_COLOR = MAGENTA

DEFAULT_MODEL_FILENAME = "cube_data.txt"

initial_cube_vertices_3d = np.array([
    [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]
])
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]
initial_cube_vertices_homogeneous = np.hstack([
    initial_cube_vertices_3d, np.ones((initial_cube_vertices_3d.shape[0], 1))
])

model_transform_matrix = np.identity(4)

def get_translation_matrix(dx, dy, dz):
    return np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [dx,dy,dz,1]])

def get_scaling_matrix(sx, sy, sz):
    return np.array([[sx,0,0,0], [0,sy,0,0], [0,0,sz,0], [0,0,0,1]])

def get_rotation_x_matrix(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1,0,0,0], [0,c,s,0], [0,-s,c,0], [0,0,0,1]])

def get_rotation_y_matrix(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c,0,-s,0], [0,1,0,0], [s,0,c,0], [0,0,0,1]])

def get_rotation_z_matrix(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c,s,0,0], [-s,c,0,0], [0,0,1,0], [0,0,0,1]])

def get_perspective_projection_matrix(focal_length_d):
    d = focal_length_d
    if abs(d) < 1e-5: d = np.sign(d) * 1e-5 if d != 0 else 1e-5
    return np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1/d], [0,0,0,1]])

def get_orthographic_projection_matrix():
    return np.array([[1,0,0,0], [0,1,0,0], [0,0,0,0], [0,0,0,1]])

def draw_text(surface, text, pos, font, color=WHITE, background=None):
    text_surface = font.render(text, True, color, background)
    surface.blit(text_surface, pos)

def load_model_from_file(filename):
    global initial_cube_vertices_3d, initial_cube_vertices_homogeneous, model_transform_matrix
    new_vertices_list = []
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f):
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    continue
                parts = stripped_line.split()
                if len(parts) == 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        new_vertices_list.append([x, y, z])
                    except ValueError:
                        msg = f"Ошибка: Неверный формат коорд. в '{filename}' строка {line_num + 1}."
                        print(msg)
                        return False, msg
                else:
                    msg = f"Ошибка: Не 3 элемента в '{filename}' строка {line_num + 1}."
                    print(msg)
                    return False, msg

        if len(new_vertices_list) != 8:
            msg = f"Ошибка: '{filename}' должен содержать 8 вершин. Найдено: {len(new_vertices_list)}."
            print(msg)
            return False, msg

        initial_cube_vertices_3d = np.array(new_vertices_list)
        initial_cube_vertices_homogeneous = np.hstack([
            initial_cube_vertices_3d, np.ones((initial_cube_vertices_3d.shape[0], 1))
        ])
        model_transform_matrix = np.identity(4)
        msg = f"Модель успешно загружена из '{filename}'."
        print(msg)
        return True, msg

    except FileNotFoundError:
        msg = f"Ошибка: Файл '{filename}' не найден."
        print(msg)
        return False, msg
    except Exception as e:
        msg = f"Непредвиденная ошибка загрузки: {e}"
        print(msg)
        return False, msg

def save_model_to_file(filename, vertices_3d_to_save):
    global initial_cube_vertices_3d, initial_cube_vertices_homogeneous, model_transform_matrix
    try:
        with open(filename, 'w') as f:
            if vertices_3d_to_save.shape[0] != 8:
                msg = f"Ошибка сохранения: Попытка сохранить {vertices_3d_to_save.shape[0]} вершин. Ожидалось 8."
                print(msg)
                return False, msg
            for vertex in vertices_3d_to_save:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        initial_cube_vertices_3d = np.copy(vertices_3d_to_save)
        initial_cube_vertices_homogeneous = np.hstack([
            initial_cube_vertices_3d, np.ones((initial_cube_vertices_3d.shape[0], 1))
        ])
        model_transform_matrix = np.identity(4)
        msg = f"Модель успешно сохранена в '{filename}'."
        print(msg)
        return True, msg
    except Exception as e:
        msg = f"Ошибка сохранения модели в '{filename}': {e}"
        print(msg)
        return False, msg

def main():
    global initial_cube_vertices_3d, initial_cube_vertices_homogeneous, model_transform_matrix

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("3D Редактор Куба - Нажмите H для помощи")
    font_small = pygame.font.SysFont("monospace", 15)
    font_coords = pygame.font.SysFont("monospace", 12)
    clock = pygame.time.Clock()

    projection_type = "perspective"
    PERSPECTIVE_D_TEXT = 2.0
    VIEW_SCALE = 150
    rotate_speed, scale_speed = math.radians(2), 0.05

    show_help = True
    show_world_coords = True
    show_indices = True
    
    file_op_status_message = ""

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_h: show_help = not show_help
                if event.key == pygame.K_p: projection_type = "perspective"
                if event.key == pygame.K_o: projection_type = "orthographic"
                if event.key == pygame.K_r: model_transform_matrix = np.identity(4)
                if event.key == pygame.K_c: show_world_coords = not show_world_coords
                if event.key == pygame.K_i: show_indices = not show_indices
                if event.key == pygame.K_l:
                    success, msg = load_model_from_file(DEFAULT_MODEL_FILENAME)
                    file_op_status_message = msg
                if event.key == pygame.K_k:
                    current_transformed_h = initial_cube_vertices_homogeneous @ model_transform_matrix
                    vertices_to_save = current_transformed_h[:, :3]
                    success, msg = save_model_to_file(DEFAULT_MODEL_FILENAME, vertices_to_save)
                    file_op_status_message = msg

                reflect_matrix = np.identity(4)
                if event.key == pygame.K_F1: reflect_matrix = get_scaling_matrix(1, 1, -1)
                if event.key == pygame.K_F2: reflect_matrix = get_scaling_matrix(1, -1, 1)
                if event.key == pygame.K_F3: reflect_matrix = get_scaling_matrix(-1, 1, 1)
                model_transform_matrix = model_transform_matrix @ reflect_matrix

        keys = pygame.key.get_pressed()

        current_rotation = np.identity(4)
        if keys[pygame.K_a]: current_rotation = get_rotation_y_matrix(rotate_speed)
        if keys[pygame.K_d]: current_rotation = get_rotation_y_matrix(-rotate_speed)
        if keys[pygame.K_w]: current_rotation = get_rotation_x_matrix(rotate_speed)
        if keys[pygame.K_s]: current_rotation = get_rotation_x_matrix(-rotate_speed)
        if keys[pygame.K_q]: current_rotation = get_rotation_z_matrix(rotate_speed)
        if keys[pygame.K_e]: current_rotation = get_rotation_z_matrix(-rotate_speed)
        model_transform_matrix = model_transform_matrix @ current_rotation

        current_scale = np.identity(4)
        scale_factor = 1.0
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]: scale_factor = 1 + scale_speed
        if keys[pygame.K_MINUS]: scale_factor = 1 - scale_speed

        if scale_factor != 1.0:
            current_scale = get_scaling_matrix(scale_factor, scale_factor, scale_factor)

        if keys[pygame.K_x]:
            mod_x_scale = 1 + scale_speed if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1 - scale_speed
            current_scale = current_scale @ get_scaling_matrix(mod_x_scale, 1, 1)

        model_transform_matrix = model_transform_matrix @ current_scale 

        transformed_vertices_h = initial_cube_vertices_homogeneous @ model_transform_matrix

        if projection_type == "perspective":
            projection_matrix = get_perspective_projection_matrix(PERSPECTIVE_D_TEXT)
        else:
            projection_matrix = get_orthographic_projection_matrix()

        projected_vertices_h = transformed_vertices_h @ projection_matrix

        screen_points = []
        vertex_w_values = []

        for i_vert in range(projected_vertices_h.shape[0]):
            v_h_proj = projected_vertices_h[i_vert]
            w_prime = v_h_proj[3]
            vertex_w_values.append(w_prime)

            if abs(w_prime) < 1e-5:
                screen_x, screen_y = SCREEN_WIDTH * 10, SCREEN_HEIGHT * 10
            else:
                px = v_h_proj[0] / w_prime
                py = v_h_proj[1] / w_prime
                screen_x = int(px * VIEW_SCALE + SCREEN_WIDTH / 2)
                screen_y = int(py * VIEW_SCALE + SCREEN_HEIGHT / 2)
            screen_points.append((screen_x, screen_y))

        screen.fill(BLACK)

        for edge_idx, edge in enumerate(cube_edges):
            p1_idx, p2_idx = edge
            valid_edge = True
            if projection_type == "perspective":
                epsilon = 1e-4
                if p1_idx < len(vertex_w_values) and p2_idx < len(vertex_w_values):
                    if vertex_w_values[p1_idx] <= epsilon or vertex_w_values[p2_idx] <= epsilon:
                        valid_edge = False
            
            if valid_edge and p1_idx < len(screen_points) and p2_idx < len(screen_points):
                try:
                    pygame.draw.line(screen, WHITE, screen_points[p1_idx], screen_points[p2_idx], 1)
                except IndexError:
                    pass

        for i_vert in range(len(screen_points)):
            sp_x, sp_y = screen_points[i_vert]
            if 0 <= sp_x <= SCREEN_WIDTH and 0 <= sp_y <= SCREEN_HEIGHT:
                text_offset_y = 0
                if show_indices:
                    draw_text(screen, f"{i_vert}", (sp_x + 5, sp_y + text_offset_y), font_coords, INDEX_COLOR)
                    text_offset_y += 12
                if show_world_coords:
                    if i_vert < transformed_vertices_h.shape[0]:
                        wx, wy, wz, _ = transformed_vertices_h[i_vert]
                        coord_text = f"({wx:.1f},{wy:.1f},{wz:.1f})"
                        draw_text(screen, coord_text, (sp_x + 5, sp_y + text_offset_y), font_coords, COORDS_TEXT_COLOR)
                        text_offset_y += 12

        help_lines = []
        if show_help:
            help_lines.extend([
                "--- Управление (H: Помощь вкл/выкл) ---",
                "A/D: Вращение Y | W/S: Вращение X | Q/E: Вращение Z",
                "+/-: Масштабирование | X / Shift+X: Масшт. по X",
                "F1/F2/F3: Отражение отн. пл. XY/XZ/YZ",
                "P: Перспектива | O: Ортография | R: Сброс",
                f"L: Загрузить из '{DEFAULT_MODEL_FILENAME}'",
                f"K: Сохранить в '{DEFAULT_MODEL_FILENAME}' (перезапись!)",
                "--- Переключатели отображения ---",
                "I: Индексы вершин",
                "C: Координаты (X,Y,Z до проекции)",
            ])

        projection_type_rus = {
            "perspective": "Перспективная",
            "orthographic": "Ортографическая"
        }

        status_lines = [
            f"Проекция: {projection_type_rus.get(projection_type, projection_type)} (d={PERSPECTIVE_D_TEXT:.1f})",
            f"Масштаб вида: {VIEW_SCALE}",
            f"Трансформирован: {'Да' if not np.array_equal(model_transform_matrix, np.identity(4)) else 'Нет'}",
            f"Индексы: {'Вкл' if show_indices else 'Выкл'} (I) | Координаты: {'Вкл' if show_world_coords else 'Выкл'} (C)"
        ]
        if file_op_status_message:
            status_lines.append(file_op_status_message)

        for i_line, line in enumerate(help_lines):
            draw_text(screen, line, (10, 10 + i_line * 18), font_small, INFO_TEXT_COLOR)

        start_y_status = SCREEN_HEIGHT - (len(status_lines)) * 18 - 5
        for i_line, line in enumerate(status_lines):
            draw_text(screen, line, (10, start_y_status + i_line * 18), font_small, INFO_TEXT_COLOR)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()