"""Application to explore and visualize VAE."""
import argparse
import datetime
import json
import os
import sys
import time
import trimesh

# need to import PySide2 before matplotlib
# see: https://matplotlib.org/3.2.1/gallery/user_interfaces/embedding_in_qt_sgskip.html # noqa
import ffmpeg
from PySide2.QtWidgets import (
    QApplication,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QTabWidget,
    QWidget,
    QFileDialog,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QDoubleSpinBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator
import numpy as np
from PySide2 import QtCore
import torch
import yoco

import sdfest
from sdfest.vae.sdf_vae import SDFVAE
from sdfest.vae import sdf_utils, utils


class ArgumentParserError(Exception):
    pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)


class VAEVisualizer(QDialog):
    """User interface to explore VAE model."""

    LATENT_ABS_MAX = 4
    LATENT_TICKS = 100

    def __init__(self, parent=None):
        super(VAEVisualizer, self).__init__(parent)

        # define attributes
        self._single_sdf_input = None
        self._single_sdf_latent = None
        self._single_sdf_output = None

        self._transition_sdf_first = None
        self._transition_sdf_first_latent = None
        self._transition_sdf_first_out = None
        self._transition_sdf_second = None
        self._transition_sdf_second_latent = None
        self._transition_sdf_second_out = None
        self._transition_sdf_out = None

        self._config = None
        self._state_dict = None
        self._model = None
        self._latent_sliders = []

        self._isosurface_level = 0

        # split layout into two halfs
        # (left for loading model, right is for visualization)
        layout = QHBoxLayout()
        model_layout = QVBoxLayout()

        # Load model button
        load_model_button = QPushButton("Load model/config")
        load_model_button.clicked.connect(self.load_model)
        load_model_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        model_layout.addWidget(load_model_button)

        # Confg line edit
        self._config_line_edit = QLineEdit()
        self._config_line_edit.textChanged.connect(self.config_changed)
        self._config_line_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        model_layout.addWidget(self._config_line_edit)

        # Isosurface leve spin box
        self._iso_level_spinbox = QDoubleSpinBox()
        self._iso_level_spinbox.setDecimals(2)
        self._iso_level_spinbox.setSingleStep(0.01)
        self._iso_level_spinbox.valueChanged.connect(self.update_isosurface_level)
        model_layout.addWidget(self._iso_level_spinbox)

        # Model status
        self._model_status_label = QLabel()
        model_layout.addWidget(self._model_status_label)

        # Current config
        scroll_area = QScrollArea(widgetResizable=True)
        scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        scroll_area.setMinimumWidth(270)
        self._config_label = QLabel()
        self._config_label.setAlignment(QtCore.Qt.AlignTop)
        scroll_area.setWidget(self._config_label)

        model_layout.addWidget(scroll_area)

        # Single object explorer
        single_object_widget = QWidget()
        single_object_widget_layout = QVBoxLayout()
        single_object_widget_layout.setAlignment(QtCore.Qt.AlignTop)
        single_object_widget_bottom = QWidget()
        single_object_widget_bottom_layout = QHBoxLayout()
        single_object_save_buttons_widget = QWidget()
        single_object_save_buttons_layout = QHBoxLayout()
        self.slider_group_widget = QWidget()
        self.slider_group_widget_layout = QVBoxLayout()
        load_sdf_button = QPushButton("Load SDF")
        load_sdf_button.clicked.connect(self.load_single_sdf)
        generate_sdf_button = QPushButton("Generate SDF from prior")
        generate_sdf_button.clicked.connect(self.generate_sdf)
        save_figure_button = QPushButton("Save figure")
        save_figure_button.clicked.connect(self.save_figure)
        save_mesh_button = QPushButton("Save mesh")
        save_mesh_button.clicked.connect(self.save_mesh)
        save_sdf_button = QPushButton("Save sdf")
        save_sdf_button.clicked.connect(self.save_sdf)

        self.single_object_figure = Figure(
            dpi=85, facecolor=(1, 1, 1), edgecolor=(0, 0, 0), tight_layout=True
        )
        single_object_canvas = FigureCanvas(self.single_object_figure)
        single_object_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.scroll_area_sliders = QScrollArea(widgetResizable=True)
        # widgetResizable indicates that the widget insided can be resized
        self.scroll_area_sliders.setFixedWidth(130)

        self.slider_group_widget.setLayout(self.slider_group_widget_layout)
        self.scroll_area_sliders.setWidget(self.slider_group_widget)
        self.scroll_area_sliders.show()

        # Single object bottom
        single_object_widget_bottom_layout.addWidget(single_object_canvas)
        single_object_widget_bottom_layout.addWidget(self.scroll_area_sliders)
        single_object_widget_bottom.setLayout(single_object_widget_bottom_layout)

        # Single object save buttons
        single_object_save_buttons_layout.addWidget(save_figure_button)
        single_object_save_buttons_layout.addWidget(save_mesh_button)
        single_object_save_buttons_layout.addWidget(save_sdf_button)
        single_object_save_buttons_widget.setLayout(single_object_save_buttons_layout)
        single_object_save_buttons_layout.setMargin(0)

        # Single object tab
        single_object_widget_layout.addWidget(load_sdf_button)
        single_object_widget_layout.addWidget(generate_sdf_button)
        single_object_widget_layout.addWidget(single_object_save_buttons_widget)
        single_object_widget_layout.addWidget(single_object_widget_bottom)
        single_object_widget.setLayout(single_object_widget_layout)

        # Transition widget
        transition_widget = QWidget()

        transition_widget_layout = QVBoxLayout()
        transition_widget_layout.setAlignment(QtCore.Qt.AlignTop)

        transition_widget_button_layout = QHBoxLayout()

        load_first_sdf_button = QPushButton("Load SDF 1")
        load_first_sdf_button.clicked.connect(lambda: self.load_transition_sdf("first"))

        load_second_sdf_button = QPushButton("Load SDF 2")
        load_second_sdf_button.clicked.connect(
            lambda: self.load_transition_sdf("second")
        )

        self.transition_figure = Figure(
            dpi=85, facecolor=(1, 1, 1), edgecolor=(0, 0, 0)
        )
        transition_canvas = FigureCanvas(self.transition_figure)
        transition_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.transition_figure_axes = self.transition_figure.subplots(2, 3)
        self.transition_figure.delaxes(self.transition_figure_axes[0, 1])

        self.transition_slider = QSlider(QtCore.Qt.Horizontal)
        self.transition_slider.setMinimum(0)
        self.transition_slider.setMaximum(100)
        self.transition_slider.valueChanged.connect(self.update_transition_slider)

        transition_widget_button_layout.addWidget(load_first_sdf_button)
        transition_widget_button_layout.addWidget(load_second_sdf_button)
        transition_widget_layout.addLayout(transition_widget_button_layout)
        transition_widget_layout.addWidget(transition_canvas)
        transition_widget_layout.addWidget(self.transition_slider)
        transition_widget.setLayout(transition_widget_layout)

        # Animation widget
        animation_widget = QWidget()
        animation_widget_layout = QVBoxLayout()
        animation_widget_layout.setAlignment(QtCore.Qt.AlignTop)

        add_kf_from_single_view_button = QPushButton("Add from single view")
        add_kf_from_single_view_button.clicked.connect(
            lambda: self.add_kf_from_single_view()
        )

        delete_kf_button = QPushButton("Delete keyframe")
        delete_kf_button.clicked.connect(lambda: self.delete_kf())

        generate_animation_button = QPushButton("Generate animation")
        generate_animation_button.clicked.connect(lambda: self.generate_animation())

        self._keyframe_list = QListWidget()

        animation_widget.setLayout(animation_widget_layout)
        animation_widget_layout.addWidget(add_kf_from_single_view_button)
        animation_widget_layout.addWidget(delete_kf_button)
        animation_widget_layout.addWidget(self._keyframe_list)
        animation_widget_layout.addWidget(generate_animation_button)

        # Tab widget
        tab_widget = QTabWidget()
        tab_widget.addTab(single_object_widget, "Single SDF")
        tab_widget.addTab(transition_widget, "Transition")
        tab_widget.addTab(animation_widget, "Animation")

        layout.addLayout(model_layout)
        layout.addWidget(tab_widget)
        layout.setStretch(0, 0)
        layout.setStretch(1, 2)

        self.setLayout(layout)

        self.parse_config()
        self.update_model()

        # TODO make these adjustable from GUI
        self.transform = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        self.azimuth = 0
        self.polar_angle = -np.pi / 4

    def reset_output(self) -> None:
        self._single_sdf_latent = None
        self._single_sdf_output = None
        self._transition_sdf_first_latent = None
        self._transition_sdf_first_out = None
        self._transition_sdf_second_latent = None
        self._transition_sdf_second_out = None

    def add_kf_from_single_view(self) -> None:
        """Add keyframe based on current latent in Single SDF tab."""
        if self._single_sdf_latent is not None:
            item = QListWidgetItem()
            item.setText(str(self._single_sdf_latent))
            item.setData(QtCore.Qt.UserRole, self._single_sdf_latent)
            self._keyframe_list.addItem(item)
        else:
            print("No valid latent currently set in Single SDF tab")

    def delete_kf(self) -> None:
        """Delete currently selected keyframe."""
        row = self._keyframe_list.currentRow()
        self._keyframe_list.takeItem(row)

    def generate_animation(self) -> None:
        """Generate animation based on keyframes."""
        fps = 30
        distance_per_frame = 0.1
        loop = True
        keyframes = []
        for i in range(self._keyframe_list.count()):
            keyframes.append(self._keyframe_list.item(i).data(QtCore.Qt.UserRole))
        if loop:
            keyframes.append(keyframes[0])

        # generate frames
        with torch.no_grad():
            fig = Figure(dpi=85, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
            now = datetime.datetime.now()
            folder = now.strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(folder)
            frame_number = 0
            current = keyframes.pop(0).clone()
            remaining_distance = distance_per_frame
            while keyframes:
                next_target = keyframes[0]
                delta = next_target - current
                distance = torch.linalg.norm(delta)
                direction = delta / torch.linalg.norm(delta)
                if distance > distance_per_frame:
                    current += direction * distance_per_frame
                    remaining_distance = 0
                else:
                    current = keyframes.pop(0).clone()
                    remaining_distance -= distance

                if remaining_distance <= 0:
                    # generate frame
                    sdf = self._model.decode(current)
                    mesh = sdf_utils.mesh_from_sdf(
                        sdf[0, 0].cpu().numpy(),
                        self._isosurface_level,
                        complete_mesh=True,
                    )
                    if mesh is None:
                        print("No surface boundaries in SDF.")
                    else:
                        ax = fig.subplots(1, 1)
                        sdf_utils.plot_mesh(
                            mesh,
                            plot_object=ax,
                            transform=self.transform,
                            azimuth=self.azimuth,
                            polar_angle=self.polar_angle,
                        )
                        ax.xaxis.set_major_locator(NullLocator())
                        ax.yaxis.set_major_locator(NullLocator())
                        fig.savefig(
                            os.path.join(folder, f"{frame_number:05d}.png"), dpi=200
                        )
                        frame_number += 1
                        fig.clear()

        # convert frames to video
        video_name = os.path.join(f"{folder}.mp4")
        ffmpeg.input(
            os.path.join(folder, "*.png"), pattern_type="glob", framerate=fps
        ).output(video_name).run()

    def update_model(self):
        if self._config is None:
            return
        try:
            self._model = SDFVAE(
                sdf_size=64,
                latent_size=self._config["latent_size"],
                encoder_dict=self._config["encoder"],
                decoder_dict=self._config["decoder"],
                device="cpu",
                tsdf=self._config["tsdf"],
            )

            if self._state_dict is not None:
                self._model.load_state_dict(self._state_dict)
                self._model_status_label.setText(
                    "Model (with weights) successfully loaded."
                )
            else:
                self._model_status_label.setText(
                    "Model (no weights) successfully loaded."
                )
            self._model_status_label.setStyleSheet("color: green")
            self.reset_output()
        except RuntimeError:
            self._model_status_label.setText("Model config and weights do not match.")
            self._model_status_label.setStyleSheet("color: red")
            self._config_label.setText("")
            self._model = None
        except KeyError:
            self._model_status_label.setText("Model config incomplete.")
            self._model_status_label.setStyleSheet("color: red")
            self._config_label.setText("")
            self._model = None

        self.update_single()
        self.update_transition()

    def load_sdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open file", filter="*.npy")
        if path == "":
            return None
        try:
            sdf_np = np.load(path)
            return torch.as_tensor(sdf_np).unsqueeze(0).unsqueeze(0)
        except FileNotFoundError:
            print("Error loading SDF.")
            return None

    def parse_config(self):
        try:
            arg_list = self._config_line_edit.text().split(" ")
            arg_list = [x for x in arg_list if x != ""]
            parser = ThrowingArgumentParser()
            parser.add_argument("--tsdf", type=utils.str_to_tsdf, default=False)
            parser.add_argument("--latent_size", type=lambda x: int(float(x)))
            parser.add_argument("--config", default="configs/default.yaml", nargs="+")
            self._config = yoco.load_config_from_args(parser, arg_list)
            self._config_label.setText(json.dumps(self._config, indent=4))
        except ArgumentParserError:
            self._model_status_label.setText("Error in config string.")
            self._model_status_label.setStyleSheet("color: red")
            self._config_label.setText("")
            self._model = None
            self._config = None
        except (FileNotFoundError, IsADirectoryError):
            self._model_status_label.setText("Specified config file not found.")
            self._model_status_label.setStyleSheet("color: red")
            self._config_label.setText("")
            self._model = None
            self._config = None
        except RuntimeError:
            self._model_status_label.setText("Model config and weights do not match.")
            self._model_status_label.setStyleSheet("color: red")
            self._config_label.setText("")
            self._model = None
            self._config = None

    def load_transition_sdf(self, which):
        sdf = self.load_sdf()
        if which == "first":
            self._transition_sdf_first = sdf
            self._transition_sdf_first_out = None
            self._transition_sdf_first_latent = None
        elif which == "second":
            self._transition_sdf_second = sdf
            self._transition_sdf_second_out = None
            self._transition_sdf_second_latent = None
        else:
            raise ValueError("which must be either 'first' or 'second'.")
        self.update_transition()

    def update_single(self, evaluate=True):
        # Visualize input sdf
        self.single_object_figure.clear()
        if self._single_sdf_input is not None:
            ax1, ax2 = self.single_object_figure.subplots(1, 2)
            self.render_sdf(ax1, self._single_sdf_input[0, 0].numpy())
        else:
            ax2 = self.single_object_figure.subplots(1, 1)

        if self._model is not None:
            with torch.no_grad():
                if (
                    self._single_sdf_input is not None
                    and self._single_sdf_latent is None
                ):
                    inp = self._single_sdf_input.clone()
                    self._model.prepare_input(inp)
                    self._single_sdf_latent, _, _ = self._model.encode(inp)
                    self.update_sliders()
                if self._single_sdf_latent is not None:
                    if evaluate:
                        t1 = time.time()
                        self._single_sdf_output = self._model.decode(
                            self._single_sdf_latent, enforce_tsdf=True
                        )
                        print(f"Decoding took {time.time() - t1}s")
                    t1 = time.time()
                    self.render_sdf(ax2, self._single_sdf_output[0, 0].detach().numpy())
                    print(f"Rendering took {time.time() - t1}s")

        self.single_object_figure.canvas.draw()

    def update_sliders(self):
        for current_slider in self._latent_sliders:
            self.slider_group_widget_layout.removeWidget(current_slider)
            current_slider.deleteLater()

        self._latent_sliders = []
        for latent in self._single_sdf_latent.view(-1).numpy():
            slider = QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(-self.LATENT_ABS_MAX * self.LATENT_TICKS / 2.0)
            slider.setMaximum(+self.LATENT_ABS_MAX * self.LATENT_TICKS / 2.0)
            slider.setValue(latent * self.LATENT_TICKS / 2.0)
            slider.valueChanged.connect(self.update_latent)
            self._latent_sliders.append(slider)
            self.slider_group_widget_layout.addWidget(slider)

    def update_latent(self):
        for i, current_slider in enumerate(self._latent_sliders):
            self._single_sdf_latent[0, i] = (
                current_slider.value() * 2.0 / self.LATENT_TICKS
            )
        self.update_single()

    def update_transition_slider(self):
        self.update_transition()

    def update_transition(self, update_iso_level_only=False):
        if update_iso_level_only:
            if self._transition_sdf_first_out is not None:
                self.render_sdf(
                    self.transition_figure_axes[1, 0],
                    self._transition_sdf_first_out[0, 0].detach().numpy(),
                )
            if self._transition_sdf_second_out is not None:
                self.render_sdf(
                    self.transition_figure_axes[1, 2],
                    self._transition_sdf_second_out[0, 0].detach().numpy(),
                )
            if self._transition_sdf_out is not None:
                self.render_sdf(
                    self.transition_figure_axes[1, 1],
                    self._transition_sdf_out[0, 0].detach().numpy(),
                )
            self.transition_figure.canvas.draw()
            return

        with torch.no_grad():
            if (
                self._transition_sdf_first is not None
                and self._transition_sdf_first_out is None
            ):
                print("render first sdf")
                self.render_sdf(
                    self.transition_figure_axes[0, 0],
                    self._transition_sdf_first[0, 0].detach().numpy(),
                )
                if (
                    self._model is not None
                    and self._transition_sdf_first_latent is None
                ):
                    inp = self._transition_sdf_first.clone()
                    self._model.prepare_input(inp)
                    self._transition_sdf_first_latent, _, _ = self._model.encode(inp)
                    self._transition_sdf_first_out = self._model.decode(
                        self._transition_sdf_first_latent
                    )
                    self.render_sdf(
                        self.transition_figure_axes[1, 0],
                        self._transition_sdf_first_out[0, 0].detach().numpy(),
                    )
            if (
                self._transition_sdf_second is not None
                and self._transition_sdf_second_out is None
            ):
                self.render_sdf(
                    self.transition_figure_axes[0, 2],
                    self._transition_sdf_second[0, 0].detach().numpy(),
                )
                if (
                    self._model is not None
                    and self._transition_sdf_second_latent is None
                ):
                    inp = self._transition_sdf_second.clone()
                    self._model.prepare_input(inp)
                    self._transition_sdf_second_latent, _, _ = self._model.encode(inp)
                    self._transition_sdf_second_out = self._model.decode(
                        self._transition_sdf_second_latent
                    )
                    self.render_sdf(
                        self.transition_figure_axes[1, 2],
                        self._transition_sdf_second_out[0, 0].detach().numpy(),
                    )

            if (
                self._transition_sdf_first_latent is not None
                and self._transition_sdf_second_latent is not None
            ):
                t = self.transition_slider.value() / 100.0
                interpolated_latent = (1 - t) * self._transition_sdf_first_latent + (
                    t
                ) * self._transition_sdf_second_latent
                self._transition_sdf_out = self._model.decode(interpolated_latent)
                self.render_sdf(
                    self.transition_figure_axes[1, 1],
                    self._transition_sdf_out[0, 0].detach().numpy(),
                )
        self.transition_figure.canvas.draw()

    def load_single_sdf(self):
        sdf = self.load_sdf()
        self._single_sdf_input = sdf
        self._single_sdf_latent = None
        self.update_single()

    def generate_sdf(self):
        self._single_sdf_input = None
        if self._model is not None:
            self._single_sdf_latent = self._model.sample(1) * 0.5
            self.update_single()
            self.update_sliders()
        else:
            print("Can't generate without loaded model.")

    def save_figure(self):
        """Save current single sample figure."""
        now = datetime.datetime.now()
        filename = now.strftime("%Y-%m-%d_%H-%M-%S.png")
        self.single_object_figure.savefig(filename, pad_inches=0, dpi=200)
        print(f"Saved as {filename}")

    def save_mesh(self):
        """Save current object mesh."""
        if self._single_sdf_output is None:
            print("Can't save mesh without generated SDF.")
            return
        sdf = self._single_sdf_output[0, 0].detach().numpy()
        mesh = sdf_utils.mesh_from_sdf(sdf, self._isosurface_level, complete_mesh=True)
        mesh_filename = (
            f"mesh_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.obj"
        )
        mesh_path = os.path.join(os.getcwd(), mesh_filename)
        with open(mesh_path, "w") as f:
            f.write(trimesh.exchange.obj.export_obj(mesh))

    def save_sdf(self):
        """Save current object sdf."""
        if self._single_sdf_output is None:
            print("Can't save SDF without generated SDF.")
            return
        sdf = self._single_sdf_output[0, 0].detach().numpy()
        sdf_filename = (
            f"sdf_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.npy"
        )
        sdf_path = os.path.join(os.getcwd(), sdf_filename)
        np.save(sdf_path, sdf)

    def config_changed(self, string):
        self.parse_config()
        self.update_model()

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open file")
        if path.endswith(".yml") or path.endswith(".yaml"):
            self._config_line_edit.setText(f"--config {path}")
            self.parse_config()
            print(self._config)
            if "model" in self._config:
                self._state_dict = torch.load(self._config["model"], map_location="cpu")
            self.update_model()
        elif path.endswith(".pt"):
            self._state_dict = torch.load(path, map_location="cpu")
            self.update_model()
        else:
            return

    def render_sdf(self, ax, sdf):
        mesh = sdf_utils.mesh_from_sdf(sdf, self._isosurface_level, complete_mesh=True)
        if mesh is None:
            print("No surface boundaries in SDF.")
        else:
            sdf_utils.plot_mesh(
                mesh,
                plot_object=ax,
                transform=self.transform,
                azimuth=self.azimuth,
                polar_angle=self.polar_angle,
            )

    def update_isosurface_level(self, d):
        self._isosurface_level = d
        self.update_single(False)
        self.update_transition(True)


def main():
    """Start the UI."""
    app = QApplication(sys.argv)

    gui = VAEVisualizer()
    gui.show()

    app.exec_()


if __name__ == "__main__":
    main()
