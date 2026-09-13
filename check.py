import glob
import matplotlib.pyplot as plt
from main import load_gray, shi_tomasi_corners, lucas_kanade_track, track_with_fb_check
import numpy as np
from PIL import Image, ImageDraw

#=============================================================

img_gray = load_gray("image_00/25a73ae7f3df9a32a3311b965c06e7a8.jpg")

corners = shi_tomasi_corners(
    img_gray, 
    max_corners=200, 
    quality_level=0.01, 
    min_distance=10,
    block_size=5
)

print(f"Найдено углов: {len(corners)}")
print(f"Форма массива: {corners.shape}")

plt.figure(figsize=(12, 6))
plt.imshow(img_gray, cmap='gray')

plt.scatter(corners[:, 0], corners[:, 1], c='red', s=15, marker='x')
plt.title(f"Shi-Tomasi Corners (всего: {len(corners)})")
plt.axis('off')
plt.show()

# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from main import load_gray, shi_tomasi_corners, track_with_fb_check

img0 = load_gray("image_00/data/0000000012.png")
img1 = load_gray("image_00/data/0000000013.png")

pts0 = shi_tomasi_corners(img0, max_corners=100)

# track_with_fb_check вместо lucas_kanade_track: точка трекается вперёд,
# потом результат трекается обратно — если она не вернулась туда, откуда
# начала (с точностью до fb_threshold пикселей), это выброс
pts1, status = track_with_fb_check(img0, img1, pts0, fb_threshold=1.0)

valid_pts0 = pts0[status]
valid_pts1 = pts1[status]
rejected_pts0 = pts0[~status]

print(f"Найдено углов: {len(pts0)}")
print(f"Прошли проверку: {status.sum()}")
print(f"Отбраковано как выбросы: {(~status).sum()}")

fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.15)

current_frame = [0]

im = ax.imshow(img0, cmap='gray')
scat = ax.scatter(valid_pts0[:, 0], valid_pts0[:, 1], c='red', s=30, edgecolors='white')
# отдельно показываем отбракованные точки, чтобы видеть, что именно отсеялось
rejected_scat = ax.scatter(rejected_pts0[:, 0], rejected_pts0[:, 1],
                            c='blue', s=20, marker='o', alpha=0.6)
title = ax.set_title("Кадр k-1 (исходные точки, серым — отбракованные)", fontsize=14)
ax.axis('off')

def toggle(event=None):
    if current_frame[0] == 0:
        im.set_data(img1)
        scat.set_offsets(valid_pts1)
        scat.set_color('lime')
        rejected_scat.set_offsets(np.empty((0, 2)))  # на втором кадре не показываем
        title.set_text("Кадр k (отслеженные точки, прошедшие проверку)")
        current_frame[0] = 1
    else:
        im.set_data(img0)
        scat.set_offsets(valid_pts0)
        scat.set_color('red')
        rejected_scat.set_offsets(rejected_pts0)
        title.set_text("Кадр k-1 (исходные точки, серым — отбракованные)")
        current_frame[0] = 0
    fig.canvas.draw_idle()

def on_key(event):
    if event.key == ' ':
        toggle()

fig.canvas.mpl_connect('key_press_event', on_key)

ax_button = plt.axes([0.4, 0.03, 0.2, 0.06])
btn = Button(ax_button, 'Переключить кадр (Пробел)')
btn.on_clicked(toggle)

plt.show()