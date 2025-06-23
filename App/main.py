import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPainter, QImage
from PyQt5.QtCore import Qt, pyqtSignal
import zipfile
import os
import shutil
from apply_deep_learning import model_process
from draw_boxes import draw_boxes

class ImageWidget(QWidget):
    def __init__(self, imagePath=None, parent=None):
        super().__init__(parent)
        self.imagePath = imagePath
        self.image = None
        if imagePath:
            self.loadImage(imagePath)

    def loadImage(self, imagePath):
        self.image = QImage(imagePath)
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        qp = QPainter(self)
        if self.image:
            scaled_img = self.image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            qp.drawImage(self.rect(), scaled_img)

class MainWindow(QMainWindow):
    asyncProcessImageSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.currentImagePath = None  # Variable to store the current image path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CellVisionAI')
        self.setGeometry(100, 100, 800, 800)
        layout = QVBoxLayout()

        btnLoad = QPushButton('Load Image')
        btnLoad.clicked.connect(self.loadImage)
        layout.addWidget(btnLoad)

        btnSave = QPushButton('Save Results')
        btnSave.clicked.connect(self.onSaveResults)
        layout.addWidget(btnSave)

        self.imageWidget = ImageWidget()
        layout.addWidget(self.imageWidget, 1)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.statusBar()

        self.asyncProcessImageSignal.connect(self.processImage, Qt.QueuedConnection)
        

    def loadImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;JPEG Files (*.jpeg);;PNG Files (*.png)", options=options)
        if fileName:
            self.currentImagePath = fileName 
            # self.processImageAsync(fileName)
            self.processImage()

    def processImage(self):
        model_process(self.currentImagePath)
        base_path = os.path.splitext(self.currentImagePath)[0]
        text_file_path = base_path + '.txt'
        output_path = base_path + '_annotated' + '.png'
        draw_boxes(self.currentImagePath, text_file_path, output_path)
        self.imageWidget.loadImage(output_path)  

    def onSaveResults(self):
        if self.currentImagePath:
            base_path = os.path.splitext(self.currentImagePath)[0]  
            text_file_path = base_path + '.txt'  
            segmentation_folder = 'segmentation_masks'  
            self.saveResultsToZip(segmentation_folder, [text_file_path], f'{base_path}.zip')
            self.deleteProcessedItems(segmentation_folder, [text_file_path])

    def saveResultsToZip(self, folderPath, additionalFiles, outputZip):
        try:
            with zipfile.ZipFile(outputZip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(folderPath):
                    for file in files:
                        filePath = os.path.join(root, file)
                        arcname = os.path.relpath(filePath, start=os.path.dirname(folderPath))  
                        zipf.write(filePath, arcname)
                for file in additionalFiles:
                    if os.path.exists(file):  
                        arcname = os.path.basename(file) 
                        zipf.write(file, arcname)
            self.statusBar().showMessage('Results saved to ' + outputZip + ' successfully.')
        except Exception as e:
            self.statusBar().showMessage('Failed to save ZIP file: ' + str(e))

    def deleteProcessedItems(self, folderPath, files):
        try:
            shutil.rmtree(folderPath)
            for file in files:
                os.remove(file)
            print("Processed items have been deleted.")
        except Exception as e:
            print("Error deleting processed items:", e)

app = QApplication(sys.argv)
ex = MainWindow()
ex.show()
sys.exit(app.exec_())
