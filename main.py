import matplotlib
matplotlib.use('TkAgg')  # Используйте TkAgg для работы с окнами
from matplotlib import pyplot as plt
from PIL import Image
import torch 
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d

# Загрузка модели
attribute_selector = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# Загрузка изображения
image = Image.open("X-RAY.jpg")
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

# Обработка изображения
inputs = attribute_selector(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

pad = 16
output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad: -pad, pad: -pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

# Включите интерактивный режим
plt.ion()

# Визуализация с использованием matplotlib
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.show()

# Подготовка к визуализации в Open3D
depth_image = (output * 255 / np.max(output)).astype('uint8')
image = np.array(image)

depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(new_width, new_height, 500, 500, new_width/2, new_height/2)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# Визуализация в Open3D с редактированием
if pcd.is_empty():
    print("Point cloud is empty!")
else:
    o3d.visualization.draw_geometries_with_editing([pcd])  # Позволяет редактировать и вращать

# Дождитесь закрытия окна Matplotlib перед выходом
plt.ioff()  # Отключаем интерактивный режим
plt.show()  # Показать, если окно Matplotlib все еще открыто
