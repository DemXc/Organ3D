import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, QtCore
from PIL import Image
import torch 
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth Estimation and 3D Visualization")
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.load_button = QtWidgets.QPushButton("Load Image")
        self.visualize_button = QtWidgets.QPushButton("Visualize Depth")
        layout.addWidget(self.load_button)
        layout.addWidget(self.visualize_button)
        self.load_button.clicked.connect(self.load_image)
        self.visualize_button.clicked.connect(self.visualize_depth)
        self.image = None
        self.predicted_depth = None

    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.image = Image.open(file_name)
            self.process_image()

    def process_image(self):
        new_height = 480 if self.image.height > 480 else self.image.height
        new_height -= (new_height % 32)
        new_width = int(new_height * self.image.width / self.image.height)
        diff = new_width % 32
        new_width = new_width - diff if diff < 16 else new_width + 32 - diff
        new_size = (new_width, new_height)
        self.image = self.image.resize(new_size)
        attribute_selector = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
        model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
        inputs = attribute_selector(images=self.image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            self.predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy() * 1000.0

    def visualize_depth(self):
        if self.predicted_depth is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Load an image first!")
            return
        pad = 16
        output = self.predicted_depth[pad: -pad, pad: -pad]
        image_cropped = self.image.crop((pad, pad, self.image.width - pad, self.image.height - pad))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image_cropped)
        ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[1].imshow(output, cmap='plasma')
        ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.tight_layout()
        plt.show()
        depth_image = (output * 255 / np.max(output)).astype('uint8')
        image_np = np.array(image_cropped)
        depth_o3d = o3d.geometry.Image(depth_image)
        image_o3d = o3d.geometry.Image(image_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics(image_np.shape[1], image_np.shape[0], 500, 500, image_np.shape[1]/2, image_np.shape[0]/2)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
        print(f"Point cloud created with {len(pcd.points)} points.")
        if pcd.is_empty():
            print("Point cloud is empty!")
        else:
            o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
