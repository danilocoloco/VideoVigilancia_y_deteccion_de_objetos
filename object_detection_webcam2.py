import numpy as np
import os,time
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())
	print ('Download complete')
else:
	print ('Model already exists')

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: #aqui le da el modelo de datos
    serialized_graph = fid.read() #lee
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)#estaba true
category_index = label_map_util.create_category_index(categories)

#intializing the web camera device

import cv2
cap = cv2.VideoCapture(1)

# Llamada al método
fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)

# Deshabilitamos OpenCL, si no hacemos esto no funciona
cv2.ocl.setUseOpenCL(False)
tamanoMax = 768
movimiento = False
count = 0
contadorMovimiento=0
# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   ret = True
   while (1):
      if not ret:
        break

      ret, image_np=cap.read()


      if ret == True:
          image_np = cv2.resize(image_np, (1366, tamanoMax))
          gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
          gray = cv2.bilateralFilter(gray, 11, 17, 17)
          blur = cv2.blur(gray, (10, 10))

          fgmask = fgbg.apply(blur)  # Aplicamos la mascara
          # print("Mascara")
          columnas = cv2.reduce(fgmask, 0, cv2.REDUCE_MAX)
          filas = cv2.reduce(fgmask, 1, cv2.REDUCE_MAX)
          for x in range(len(columnas[0])):
              if columnas[0][x] > 0:
                  movimiento = True
                  # print(columnas[0][x])
                  break
          if movimiento == False:
              for x in range(len(filas)):
                  if filas[x, 0] > 0:
                      # print(filas[x,0])
                      movimiento = True
                      break

          # print(movimiento)

          if movimiento == True:
              contadorMovimiento += 1
              count += 1

              if count % 15 == 0:
                  print("movimiento a la hora " + time.strftime("%y%m%d-%H:%M:%S"))
                  nombre = "namepattern-" + time.strftime("%y-%m-%d- %H:%M:%S") + ".jpg"
                  path = "/" + nombre
                  cv2.imwrite(path, image_np)

          else:
              count = 0
              contadorMovimiento = 0
              # print("no hay movimiento")

          if count < 0:  # > 3:
              os.system(
                  "sudo ffmpeg -r 1/5 -i /path/where/imgages/are/saving/with_namepatter-*.jpg -r 30 -pix_fmt yuv420p /path/where/vid/are/going/to/save/nameofvid.mp4")
              for filename in os.listdir("."):
                  if filename.startswith("outvideo"):
                      name = "videoname" + time.strftime("%y%m%d-%H:%M:%S") + ".mp4"
                      os.rename(filename, name)
              count = 0

          movimiento = False

      cv2.imshow('Imagen',image_np)
      #cv2.imshow('WebCam. Para cerrar pulse la tecla "s"',cv2.resize(image_np,(1680,1066)))
      k = cv2.waitKey(30) & 0xff
      if k == ord("s"):
          ret=False
          break

   cap.release()
   cv2.destroyAllWindows()


