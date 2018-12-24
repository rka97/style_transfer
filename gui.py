import sys
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon, QImage, QRegExpValidator, qRgb
from PyQt5.QtCore import pyqtSlot, QRegExp

from UI import UI
from main import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class App():
    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)

        self.gray_color_table = [qRgb(i, i, i) for i in range(256)]

        # self.int_validator = QRegExpValidator(QRegExp("^([+-]?[0-9]\d*|0)$"))
        self.uint_validator = QRegExpValidator(QRegExp("^([+]?[0-9]\d*|0)$"))
        self.glevel_validator = QRegExpValidator(QRegExp("\\b(1?[0-9]{1,2}|2[0-4][0-9]|25[0-5])\\b"))
        self.ratio_validator = QRegExpValidator(QRegExp("0+([.][0-9]+)?|1([.]0)?"))
        self.float_validator = QRegExpValidator(QRegExp("[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)"))

        self.ui = UI(title="Style Transfer")
        self.ui.transfer_btn.clicked.connect(self.transfer)
        self.ui.original_browse_btn.clicked.connect(self.set_original_image)
        self.ui.stlye_browse_btn.clicked.connect(self.set_style_image)
        self.ui.export_btn.clicked.connect(self.export)
        self.ui.get_mask_btn.clicked.connect(self.get_segmentation_mask)
        self.ui.segmentation_mode_combo.currentTextChanged.connect(self.set_stack_view)
        self.ui.grab_cut_mode_combo.currentTextChanged.connect(self.set_grab_cut_mode)
        self.ui.cv_init_level_set_combo.currentTextChanged.connect(self.set_chan_vese_init_level)
        self.ui.mcv_init_level_set_combo.currentTextChanged.connect(self.set_morphological_chan_vese_init_level)
        self.ui.fs_mcv_mode_combo.currentTextChanged.connect(self.set_fs_morphological_chan_vese_mode)

        self.set_validators()

        self.content_image = -1
        self.style_image = -1
        self.output_image = False
        self.grab_cut_mode = cv2.GC_INIT_WITH_MASK
        self.fs_morphological_chan_vese_init_level = "edges"
        self.chan_vese_init_level = "checkerboard"
        self.morphological_chan_vese_init_level = "edges"
        self.x = None
        self.c = None
        self.mask = None

    def run(self):
        sys.exit(self.app.exec_())

    def transfer(self):
        if self.content_image == -1 or self.style_image == -1:
            return

        self.get_segmentation_mask()
        self.x = main_gui(self.content_image, self.style_image, self.mask)
        self.output_image = True
        width = self.ui.output_image_view.width()
        height = self.ui.output_image_view.height()

        h, w, channel = self.x.shape
        bytes_per_line = 3 * w
        image = QImage(self.x.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QtGui.QPixmap(image)
        pix = pix.scaled(width, height, QtCore.Qt.KeepAspectRatio)

        self.ui.output_image_view.setPixmap(pix)

    def export(self):
        filename = self.save_file_name_dialog()

        if not self.output_image or filename == -1:
            return

        io.imsave(filename, self.x)

    def set_grab_cut_mode(self):
        index = self.ui.grab_cut_mode_combo.currentIndex()

        if index == 0:
            self.grab_cut_mode = cv2.GC_INIT_WITH_MASK
        elif index == 1:
            self.grab_cut_mode = cv2.GC_INIT_WITH_RECT
        elif index == 2:
            self.grab_cut_mode = cv2.GC_EVAL

    def set_chan_vese_init_level(self):
        index = self.ui.cv_init_level_set_combo.currentIndex()

        if index == 0:
            self.chan_vese_init_level = "checkerboard"
        elif index == 1:
            self.chan_vese_init_level = "disk"
        elif index == 2:
            self.chan_vese_init_level = "small disk"
        elif index == 3:
            self.chan_vese_init_level = "edges"
        elif index == 4:
            self.chan_vese_init_level = "original gray"
        elif index == 5:
            image_file = self.open_file_name_dialog()
            if image_file == -1:
                return
            self.chan_vese_init_level = "path," + image_file

    def set_morphological_chan_vese_init_level(self):
        index = self.ui.mcv_init_level_set_combo.currentIndex()

        if index == 0:
            self.morphological_chan_vese_init_level = "edges"
        elif index == 1:
            self.morphological_chan_vese_init_level = "checkerboard"
        elif index == 2:
            self.morphological_chan_vese_init_level = "circle"
        elif index == 3:
            self.morphological_chan_vese_init_level = "original gray"
        elif index == 4:
            image_file = self.open_file_name_dialog()
            if image_file == -1:
                return
            self.morphological_chan_vese_init_level = "path," + image_file

    def set_fs_morphological_chan_vese_mode(self):
        index = self.ui.fs_mcv_mode_combo.currentIndex()

        if index == 0:
            self.fs_morphological_chan_vese_init_level = "edges"
        elif index == 1:
            self.fs_morphological_chan_vese_init_level = "checkerboard"
        elif index == 2:
            self.fs_morphological_chan_vese_init_level = "circle"
        elif index == 3:
            self.fs_morphological_chan_vese_init_level = "original gray"
        elif index == 4:
            image_file = self.open_file_name_dialog()
            if image_file == -1:
                return
            self.fs_morphological_chan_vese_init_level = "path," + image_file

    def set_stack_view(self):
        self.ui.stackedWidget.setCurrentIndex(self.ui.segmentation_mode_combo.currentIndex())

    def set_validators(self):
        # general mask parameters
        self.ui.mask_c_input.setValidator(self.float_validator)
        # face segmentatin
        self.ui.scale_factor_input.setValidator(self.float_validator)
        self.ui.min_neighbours_input.setValidator(self.uint_validator)
        self.ui.canny_sigma_input.setValidator(self.float_validator)
        self.ui.mcv_gaussian_sigma_input.setValidator(self.float_validator)
        self.ui.canny_low_threshold_input.setValidator(self.ratio_validator)
        self.ui.canny_high_threshold_input.setValidator(self.ratio_validator)
        self.ui.num_dialation_input.setValidator(self.uint_validator)
        self.ui.fs_mcv_c1_input.setValidator(self.float_validator)
        self.ui.fs_mcv_c2_input.setValidator(self.float_validator)
        self.ui.fs_mcv_num_iter_input.setValidator(self.uint_validator)
        self.ui.fs_mcv_smoothing_input.setValidator(self.uint_validator)
        self.ui.fs_mcv_threshold_input.setValidator(self.uint_validator)
        self.ui.fs_gaussian_sigma_input.setValidator(self.float_validator)
        self.ui.grab_cut_num_iter_input.setValidator(self.uint_validator)
        self.ui.model_size_input.setValidator(self.uint_validator)
        self.ui.fs_dialation_sigma_input.setValidator(self.float_validator)

        # edge segmentation
        self.ui.edge_strength_input.setValidator(self.float_validator)
        self.ui.edge_coherance_input.setValidator(self.float_validator)
        # convex hull
        self.ui.ch_ethreshold_input.setValidator(self.float_validator)
        # watershed
        self.ui.ws_ethreshold_input.setValidator(self.float_validator)
        self.ui.ws_mdisk_size_input.setValidator(self.uint_validator)
        self.ui.ws_mthreshold_input.setValidator(self.uint_validator)
        self.ui.ws_gdisk_size_input.setValidator(self.uint_validator)
        self.ui.ws_glevel_threshold_input.setValidator(self.glevel_validator)
        # convex hull * watershed
        self.ui.chws_ch_ethreshold_input.setValidator(self.float_validator)
        self.ui.chws_ws_ethreshold_input.setValidator(self.float_validator)
        self.ui.chws_mdisk_size_input.setValidator(self.uint_validator)
        self.ui.chws_mthreshold_input.setValidator(self.uint_validator)
        self.ui.chws_gdisk_size_input.setValidator(self.uint_validator)
        self.ui.chws_glevel_threshold_input.setValidator(self.glevel_validator)
        # chan vese
        self.ui.cv_ethreshold_input.setValidator(self.float_validator)
        self.ui.cv_mu_input.setValidator(self.float_validator)
        self.ui.cv_lamda_1_imput.setValidator(self.float_validator)
        self.ui.cv_lamda_2_imput.setValidator(self.float_validator)
        self.ui.cv_tol_input.setValidator(self.float_validator)
        self.ui.cv_max_iter_input.setValidator(self.uint_validator)
        self.ui.cv_dt_input.setValidator(self.float_validator)
        # morphological chan vese
        self.ui.mcv_c1_input.setValidator(self.float_validator)
        self.ui.mcv_c2_input.setValidator(self.float_validator)
        self.ui.mcv_max_iter_input.setValidator(self.uint_validator)
        self.ui.mcv_smoothing_input.setValidator(self.uint_validator)
        self.ui.mcv_gaussian_sigma_input.setValidator(self.float_validator)

    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.ui.window, "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg *.jepg)", options=options)
        if fileName:
            return fileName
        else:
            return -1

    def save_file_name_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self.ui.window, "QFileDialog.getSaveFileName()", "","Images (*.png *.xpm *.jpg *.jepg)", options=options)
        if fileName:
            return fileName
        else:
            return -1

    def set_original_image(self):
        image_file = self.open_file_name_dialog()
        if image_file == -1:
            return

        self.c = io.imread(image_file) / 255.0
        self.c = (cv2.resize(self.c, (IM_SIZE, IM_SIZE))).astype(np.float32)
        width = self.ui.original_image_view.width()
        height = self.ui.original_image_view.height()
        image = QtGui.QPixmap(image_file)
        image = image.scaled(width, height, QtCore.Qt.KeepAspectRatio)

        self.ui.original_image_view.setPixmap(image)
        self.content_image = image_file

    def set_style_image(self):
        image_file = self.open_file_name_dialog()
        if image_file == -1:
            return

        width = self.ui.style_image_view.width()
        height = self.ui.style_image_view.height()
        image = QtGui.QPixmap(image_file)
        image = image.scaled(width, height, QtCore.Qt.KeepAspectRatio)

        self.ui.style_image_view.setPixmap(image)
        self.style_image = image_file

    def convex_hull_mask(self, edge_strength, edge_coherance):
        ch_ethreshold = 0.8 if self.ui.ch_ethreshold_input.text() == "" else float(self.ui.ch_ethreshold_input.text())
        return edge_segmentation(
            self.c, mode=0, strength_threshold=edge_strength, coherence_threshold=edge_coherance,
            ch_ethreshold=ch_ethreshold
        )

    def watershed_mask(self, edge_strength, edge_coherance):
        ws_ethreshold = 0.2 if self.ui.ws_ethreshold_input.text() == "" else float(self.ui.ws_ethreshold_input.text())
        ws_mdisk_size = 5 if self.ui.ws_mdisk_size_input.text() == "" else int(self.ui.ws_mdisk_size_input.text())
        ws_mthreshold = 20 if self.ui.ws_mthreshold_input.text() == "" else int(self.ui.ws_mthreshold_input.text())
        ws_gdisk_size = 2 if self.ui.ws_gdisk_size_input.text() == "" else int(self.ui.ws_gdisk_size_input.text())
        ws_glevel_threshold = 4 if self.ui.ws_glevel_threshold_input.text() == "" else int(self.ui.ws_glevel_threshold_input.text())

        return edge_segmentation(
            self.c, mode=1, strength_threshold=edge_strength, coherence_threshold=edge_coherance,
            ws_ethreshold=ws_ethreshold, ws_mdisk_size=ws_mdisk_size, ws_mthreshold=ws_mthreshold, ws_gdisk_size=ws_gdisk_size, ws_glevel_threshold=ws_glevel_threshold
        )

    def convex_hull_watershed_mask(self, edge_strength, edge_coherance):
        ch_ethreshold = 0.8 if self.ui.ch_ethreshold_input.text() == "" else float(self.ui.ch_ethreshold_input.text())
        ws_ethreshold = 0.2 if self.ui.ws_ethreshold_input.text() == "" else float(self.ui.ws_ethreshold_input.text())
        ws_mdisk_size = 5 if self.ui.ws_mdisk_size_input.text() == "" else int(self.ui.ws_mdisk_size_input.text())
        ws_mthreshold = 20 if self.ui.ws_mthreshold_input.text() == "" else int(self.ui.ws_mthreshold_input.text())
        ws_gdisk_size = 2 if self.ui.ws_gdisk_size_input.text() == "" else int(self.ui.ws_gdisk_size_input.text())
        ws_glevel_threshold = 4 if self.ui.ws_glevel_threshold_input.text() == "" else int(self.ui.ws_glevel_threshold_input.text())

        return edge_segmentation(
            self.c, mode=2, strength_threshold=edge_strength, coherence_threshold=edge_coherance,
            ch_ethreshold=ch_ethreshold,
            ws_ethreshold=ws_ethreshold, ws_mdisk_size=ws_mdisk_size, ws_mthreshold=ws_mthreshold, ws_gdisk_size=ws_gdisk_size, ws_glevel_threshold=ws_glevel_threshold
        )

    def chan_vese_mask(self, edge_strength, edge_coherance):
        cv_ethreshold = 0 if self.ui.cv_ethreshold_input.text() == "" else float(self.ui.cv_ethreshold_input.text())
        cv_mu = 0.1 if self.ui.cv_mu_input.text() == "" else float(self.ui.cv_mu_input.text())
        cv_lamda_1 = 0.06 if self.ui.cv_lamda_1_imput.text() == "" else float(self.ui.cv_lamda_1_imput.text())
        cv_lamda_2 = 1 if self.ui.cv_lamda_2_imput.text() == "" else float(self.ui.cv_lamda_2_imput.text())
        cv_tol = 1e-3 if self.ui.cv_tol_input.text() == "" else float(self.ui.cv_tol_input.text())
        cv_max_iter = 2000 if self.ui.cv_max_iter_input.text() == "" else int(self.ui.cv_max_iter_input.text())
        cv_dt = 0.52 if self.ui.cv_dt_input.text() == "" else float(self.ui.cv_dt_input.text())
        cv_init_level_set = self.chan_vese_init_level
        cv_extended_output = False if self.ui.cv_extended_output_check.checkState() == 0 else True

        return edge_segmentation(
            self.c, mode=3, strength_threshold=edge_strength, coherence_threshold=edge_coherance,
            cv_ethreshold=cv_ethreshold, cv_mu=cv_mu, cv_lamda_1=cv_lamda_1, cv_lamda_2=cv_lamda_2, cv_tol=cv_tol, cv_max_iter=cv_max_iter, cv_dt=cv_dt, cv_init_level_set=cv_init_level_set, cv_extended_output=cv_extended_output
        )

    def morphological_chan_vese_mask(self, edge_strength, edge_coherance):
        mcv_init_level_set = self.morphological_chan_vese_init_level
        mcv_c1 = 1.0 if self.ui.mcv_c1_input.text() == "" else float(self.ui.mcv_c1_input.text())
        mcv_c2 = 1.0 if self.ui.mcv_c2_input.text() == "" else float(self.ui.mcv_c2_input.text())
        mcv_max_iter = 35 if self.ui.mcv_max_iter_input.text() == "" else int(self.ui.mcv_max_iter_input.text())
        mcv_smoothing = 1 if self.ui.mcv_smoothing_input.text() == "" else int(self.ui.mcv_smoothing_input.text())
        mcv_sigma = 5 if self.ui.mcv_gaussian_sigma_input.text() == "" else float(self.ui.mcv_gaussian_sigma_input.text())

        return edge_segmentation(
            self.c, strength_threshold=edge_strength, coherence_threshold=edge_coherance,
            mcv_c1=mcv_c1, mcv_c2=mcv_c2, mcv_init_level_set=mcv_init_level_set, mcv_max_iter=mcv_max_iter, mcv_smoothing=mcv_smoothing, mcv_sigma=mcv_sigma
        )

    def get_segmentation_mask(self):
        if self.c is None:
            QMessageBox.critical(self.ui.window, 'Error', "Can't generage Mask without content image", QMessageBox.Ok)
            return

        mode = self.ui.segmentation_mode_combo.currentIndex()
        mask_constant = 1.0 if self.ui.mask_c_input.text() == "" else float(self.ui.mask_c_input.text())
        mask = None
        if mode == 0:
            mask = get_segmentation_mask("none")
        elif mode == 1:
            scale_factor = 1.3 if self.ui.scale_factor_input.text() == "" else float(self.ui.scale_factor_input.text())
            min_neighbours = 5 if self.ui.min_neighbours_input.text() == "" else int(self.ui.min_neighbours_input.text())
            canny_sigma = 2 if self.ui.canny_sigma_input.text() == "" else float(self.ui.canny_sigma_input.text())
            mcv_gaussian_sigma = 2 if self.ui.mcv_gaussian_sigma_input.text() == "" else float(self.ui.mcv_gaussian_sigma_input.text())
            canny_low_threshold = 0.1 if self.ui.canny_low_threshold_input.text() == "" else float(self.ui.canny_low_threshold_input.text())
            canny_high_threshold = 0.2 if self.ui.canny_high_threshold_input.text() == "" else float(self.ui.canny_high_threshold_input.text())
            num_dialation = 1 if self.ui.num_dialation_input.text() == "" else int(self.ui.num_dialation_input.text())
            fs_mcv_c1 = 1.0 if self.ui.fs_mcv_c1_input.text() == "" else float(self.ui.fs_mcv_c1_input.text())
            fs_mcv_c2 = 1.0 if self.ui.fs_mcv_c2_input.text() == "" else float(self.ui.fs_mcv_c2_input.text())
            fs_mcv_num_iter = 35 if self.ui.fs_mcv_num_iter_input.text() == "" else int(self.ui.fs_mcv_num_iter_input.text())
            fs_mcv_smoothing = 1 if self.ui.fs_mcv_smoothing_input.text() == "" else int(self.ui.fs_mcv_smoothing_input.text())
            fs_mcv_threshold = 0 if self.ui.fs_mcv_threshold_input.text() == "" else int(self.ui.fs_mcv_threshold_input.text())
            fs_gaussian_sigma = 5 if self.ui.fs_gaussian_sigma_input.text() == "" else float(self.ui.fs_gaussian_sigma_input.text())
            grab_cut_num_iter = 10 if self.ui.grab_cut_num_iter_input.text() == "" else int(self.ui.grab_cut_num_iter_input.text())
            model_size = 65 if self.ui.model_size_input.text() == "" else int(self.ui.model_size_input.text())
            fs_dialtion_sigma = 2 if self.ui.fs_dialation_sigma_input.text() == "" else float(self.ui.fs_dialation_sigma_input.text())

            mask = segment_faces(
                self.c, scale_factor, min_neighbours, fs_gaussian_sigma, fs_dialtion_sigma, grab_cut_num_iter, model_size, self.grab_cut_mode,
                canny_sigma, mcv_gaussian_sigma, canny_low_threshold, canny_high_threshold, num_dialation, fs_mcv_c1, fs_mcv_c2, self.fs_morphological_chan_vese_init_level, fs_mcv_num_iter, fs_mcv_smoothing, fs_mcv_threshold
            )
        elif mode == 2:
            seg_mode = self.ui.edge_seg_algos.currentIndex()
            edge_strength = 8 if self.ui.edge_strength_input.text() == "" else float(self.ui.edge_strength_input.text())
            edge_coherance = 0.5 if self.ui.edge_coherance_input.text() == "" else float(self.ui.edge_coherance_input.text())
            if seg_mode == 0:
                mask = self.convex_hull_mask(edge_strength, edge_coherance)
            elif seg_mode == 1:
                mask = self.watershed_mask(edge_strength, edge_coherance)
            elif seg_mode == 2:
                mask = self.convex_hull_watershed_mask(edge_strength, edge_coherance)
            elif seg_mode == 3:
                mask = self.chan_vese_mask(edge_strength, edge_coherance)
            elif seg_mode == 4:
                mask = self.morphological_chan_vese_mask(edge_strength, edge_coherance)

        mask = mask * mask_constant
        self.mask = mask
        mask = mask * 255
        mask = mask.astype(np.uint8)

        width = self.ui.segmentation_mask_view.width()
        height = self.ui.segmentation_mask_view.height()
        image = QImage(mask.data, mask.shape[1], mask.shape[0], QImage.Format_Grayscale8)
        pix = QtGui.QPixmap(image)
        pix = pix.scaled(width, height, QtCore.Qt.KeepAspectRatio)

        self.ui.segmentation_mask_view.setPixmap(pix)

        QMessageBox.information(self.ui.window, 'Information', "New Mask Added", QMessageBox.Ok)

if __name__ == '__main__':
    app = App()
    app.run()
