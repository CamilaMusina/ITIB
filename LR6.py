from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QFormLayout, QGroupBox, QVBoxLayout

import numpy as np
import matplotlib.pyplot as plt
import math


class Point(object):
    def __init__(self, x, y, cluster=-1):
        self.x = x
        self.y = y
        self.cluster = cluster
        
    def euclidean_distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def chebyshev_distance(self, other):
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.cluster == other.cluster

    def __ne__(self, other):
        return not self.__eq__(other)


class ClusteriserKMeans:
    def __init__(self):
        np.random.seed = 42
        self.points = []
        self.curr_means = []
        self.cluster_number = -1
        self.output = ''
        self.distance = ''

    def add_point(self, x, y):
        self.points.append(Point(x, y))

    def add_cluster(self, x, y):
        self.curr_means.append(Point(x, y, len(self.curr_means)))

    def set_distance(self, distance):
        self.distance = distance

    def set_cluster_number(self, number):
        self.cluster_number = number

    def get_data_x_y_for_cluster(self, cl_num):
        x_cl = []
        y_cl = []
        for point in self.points:
            if point.cluster == cl_num:
                x_cl.append(point.x)
                y_cl.append(point.y)
        return x_cl, y_cl

    def get_mean_for_cluster(self, cluster):
        x_cl = []
        y_cl = []
        for point in self.points:
            if point.cluster == cluster:
                x_cl.append(point.x)
                y_cl.append(point.y)
        return np.mean(x_cl), np.mean(y_cl)

    def get_data_x_y_of_mean(self):
        x_cl = []
        y_cl = []
        for point in self.curr_means:
            x_cl.append(point.x)
            y_cl.append(point.y)
        return x_cl, y_cl

    def update_distance(self):
        for point in self.points:
            if self.distance == 'euclidean':
                distances = [point.euclidean_distance(self.curr_means[i])
                             for i in range(self.cluster_number)]
            else:
                distances = [point.chebyshev_distance(self.curr_means[i])
                             for i in range(self.cluster_number)]
            point.cluster = np.argmin(distances)

    def show_current_state(self, start=False):
        markers = ['o', 'v', 'D', 's']
        fig, ax = plt.subplots(figsize=(4, 4))
        if start:
            x_cl, y_cl = self.get_data_x_y_for_cluster(-1)
            ax.scatter(x_cl, y_cl, marker='o', label='points')
        else:
            for cluster in range(self.cluster_number):
                x_cl, y_cl = self.get_data_x_y_for_cluster(cluster)
                ax.scatter(x_cl, y_cl, marker=markers[cluster], label='cluster ' + str(cluster))
        median_x, median_y = self.get_data_x_y_of_mean()
        ax.scatter(median_x, median_y, marker="d", label='cluster centers', c='red')
        ax.grid()
        ax.legend()

    def get_clusters(self):
        iteration = 0
        self.show_current_state(True)
        plt.savefig('report' + str(iteration) + '.png')
        while True:
            prev_medians = self.curr_means.copy()

            self.update_distance()

            self.output += f'\nIteration #{iteration}\nCurrent state:\n'
            for p in self.points:
                self.output += f'{p.x}, {p.y}, {p.cluster}\n'

            for cluster in range(self.cluster_number):
                median_x, median_y = self.get_mean_for_cluster(cluster)
                self.output += f'Median value for cluster {cluster}: {format(self.curr_means[cluster].x, ".2f")}, ' \
                               f'{format(self.curr_means[cluster].y, ".2f")}\n'
                self.curr_means[cluster] = Point(median_x, median_y)

            if prev_medians == self.curr_means:
                break
            iteration += 1
            self.show_current_state()
            plt.savefig('report' + str(iteration) + '.png')

        return iteration, self.output

    def clear_clusters(self):
        for point in self.points:
            point.cluster = -1
        self.curr_means = []
        self.cluster_number = -1
        self.output = ''


class StartClusteriser(QtCore.QObject):
    start = QtCore.pyqtSignal()


class UiWidget(object):
    def __init__(self, widget):
        self.clf = ClusteriserKMeans()

        self.starter = StartClusteriser()
        self.starter.start.connect(self.clusterising)

        widget.setObjectName("Widget")
        widget.resize(800, 600)

        self.widget = QtWidgets.QWidget(widget)
        self.widget.setGeometry(QtCore.QRect(30, 40, 741, 541))
        self.widget.setObjectName("widget")

        self.xLabel = QLabel(widget)
        self.xLabel.setGeometry(QtCore.QRect(17, 40, 10, 41))
        self.xLabel.setText('x:')

        self.xLineEdit = QtWidgets.QLineEdit(widget)
        self.xLineEdit.setGeometry(QtCore.QRect(30, 40, 71, 41))
        self.xLineEdit.setObjectName("xLineEdit")

        self.yLabel = QLabel(widget)
        self.yLabel.setGeometry(QtCore.QRect(107, 40, 10, 41))
        self.yLabel.setText('y:')

        self.yLineEdit = QtWidgets.QLineEdit(widget)
        self.yLineEdit.setGeometry(QtCore.QRect(120, 40, 71, 41))
        self.yLineEdit.setObjectName("yLineEdit")

        self.AddPointButton = QtWidgets.QPushButton(widget)
        self.AddPointButton.setGeometry(QtCore.QRect(200, 40, 101, 41))
        self.AddPointButton.setObjectName("AddPointButton")
        self.AddPointButton.clicked.connect(self.on_add_point_button_clicked)

        self.AddClusterButton = QtWidgets.QPushButton(widget)
        self.AddClusterButton.setGeometry(QtCore.QRect(200, 40, 101, 41))
        self.AddClusterButton.setObjectName("AddClusterButton")
        self.AddClusterButton.hide()
        self.AddClusterButton.clicked.connect(self.on_add_cluster_button_clicked)

        self.pointsLabel = QLabel(widget)
        self.pointsLabel.setGeometry(QtCore.QRect(17, 80, 120, 41))
        self.pointsLabel.setText(f'Points number: {len(self.clf.points)}')

        self.clustersLabel = QLabel(widget)
        self.clustersLabel.setGeometry(QtCore.QRect(147, 80, 120, 41))
        self.clustersLabel.setText(f'Clusters number: {len(self.clf.curr_means)}')

        self.EuclideanButton = QtWidgets.QPushButton(widget)
        self.EuclideanButton.setGeometry(QtCore.QRect(120, 120, 80, 61))
        self.EuclideanButton.setObjectName("EuclideanButton")
        self.EuclideanButton.clicked.connect(self.on_euclidean_button_clicked)

        self.ChebyshevButton = QtWidgets.QPushButton(widget)
        self.ChebyshevButton.setGeometry(QtCore.QRect(220, 120, 80, 61))
        self.ChebyshevButton.setObjectName("ChebyshevButton")
        self.ChebyshevButton.clicked.connect(self.on_chebyshev_button_clicked)

        self.scrollArea = QtWidgets.QScrollArea(widget)
        self.scrollArea.setGeometry(QtCore.QRect(340, 40, 421, 521))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 419, 519))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.ClusterNumberLineEdit = QtWidgets.QLineEdit(widget)
        self.ClusterNumberLineEdit.setGeometry(QtCore.QRect(30, 130, 71, 41))
        self.ClusterNumberLineEdit.setObjectName("ClusterNumberLneEdit")

        self.OutputScrollArea = QtWidgets.QScrollArea(widget)
        self.OutputScrollArea.setGeometry(QtCore.QRect(30, 200, 271, 361))
        self.OutputScrollArea.setWidgetResizable(True)
        self.OutputScrollArea.setObjectName("OutputScrollArea")

        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 269, 359))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.OutputScrollArea.setWidget(self.scrollAreaWidgetContents_2)

        self.retranslate_ui(widget)
        QtCore.QMetaObject.connectSlotsByName(widget)

    def retranslate_ui(self, widget):
        _translate = QtCore.QCoreApplication.translate
        widget.setWindowTitle(_translate("Widget", "Clusteriser program"))
        self.AddPointButton.setText(_translate("Widget", "Add point"))
        self.AddClusterButton.setText(_translate("Widget", "Add cluster"))
        self.EuclideanButton.setText(_translate("Widget", "Euclidean"))
        self.ChebyshevButton.setText(_translate("Widget", "Chebyshev"))

    def on_add_point_button_clicked(self):
        x = self.xLineEdit.text()
        y = self.yLineEdit.text()
        self.clf.add_point(int(x), int(y))
        self.pointsLabel.setText(f'Points number: {len(self.clf.points)}')
        self.xLineEdit.clear()
        self.yLineEdit.clear()

    def on_add_cluster_button_clicked(self):
        x = self.xLineEdit.text()
        y = self.yLineEdit.text()
        self.clf.add_cluster(int(x), int(y))
        self.clustersLabel.setText(f'Clusters number: {len(self.clf.curr_means)}')
        self.xLineEdit.clear()
        self.yLineEdit.clear()
        if self.clf.cluster_number == len(self.clf.curr_means):
            self.AddClusterButton.setDisabled(True)
            self.starter.start.emit()

    def show_clusters(self, img_count, output):
        form_layout = QFormLayout()
        group_box = QGroupBox()

        for i in range(img_count + 1):
            label2 = QLabel()
            label2.setPixmap(QtGui.QPixmap('report' + str(i) + '.png'))
            form_layout.addRow(label2)

        group_box.setLayout(form_layout)

        self.scrollArea.setWidget(group_box)
        self.scrollArea.setWidgetResizable(True)

        layout = QVBoxLayout()
        layout.addWidget(self.scrollArea)

        container = QtWidgets.QWidget()
        self.OutputScrollArea.setWidget(container)

        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(10, 10, 0, 0)

        label = QtWidgets.QLabel(output)
        lay.addWidget(label)
        lay.addStretch()

    def on_euclidean_button_clicked(self):
        cluster_number = self.ClusterNumberLineEdit.text()
        self.clf.set_distance('euclidean')
        self.clf.set_cluster_number(int(cluster_number))
        self.EuclideanButton.setDisabled(True)
        self.ChebyshevButton.setDisabled(True)
        self.AddPointButton.hide()
        self.AddClusterButton.show()

    def on_chebyshev_button_clicked(self):
        cluster_number = self.ClusterNumberLineEdit.text()
        self.clf.set_distance('chebyshev')
        self.clf.set_cluster_number(int(cluster_number))
        self.EuclideanButton.setDisabled(True)
        self.ChebyshevButton.setDisabled(True)
        self.AddPointButton.hide()
        self.AddClusterButton.show()

    def clusterising(self):
        img_count, output = self.clf.get_clusters()
        self.show_clusters(img_count, output)


if __name__ == '__main__':
    app = QApplication([])
    window = QDialog()
    ui = UiWidget(window)

    window.show()
    app.exec()
