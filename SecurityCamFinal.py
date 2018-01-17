import cv2, time, os
import numpy as np
import os,time
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import tkinter.filedialog
import datetime
from utils import label_map_util
from utils import visualization_utils as vis_util
import logging
import string

def cargarTensorflow():
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
        print('Downloading the model')
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
        print('Download complete')
    else:
        print('Model already exists')

    # ## Load a (frozen) Tensorflow model into memory.

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:  # aqui le da el modelo de datos
            serialized_graph = fid.read()  # lee
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)  # estaba true
    category_index = label_map_util.create_category_index(categories)




def camara():

    tamanoMax = 768
    movimiento = False
    count = 0

    #file = tkinter.filedialog.askopenfilename(defaultextension="*.avi", title="Control: diropenbox",initialdir="C:/Users/DANI/Desktop/object_recognition_detection",parent=None)

    cam = cv2.VideoCapture(1)
    if (cam.isOpened() == False):
        print("ERROR: Camara no operativa")
        cam=cv2.VideoCapture(0)
        if (cam.isOpened() == False):
            exit(-1)  # Error acceso a la camara

    # Llamada al método
    fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)

    # Deshabilitamos OpenCL, si no hacemos esto no funciona
    cv2.ocl.setUseOpenCL(False)

    contadorMovimiento = 0

    ret = True

    #cargarTensorflow()

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
        print('Downloading the model')
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
        print('Download complete')
    else:
        print('Model already exists')

    # ## Load a (frozen) Tensorflow model into memory.

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:  # aqui le da el modelo de datos
            serialized_graph = fid.read()  # lee
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)  # estaba true
    category_index = label_map_util.create_category_index(categories)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("GRABACIONES/"+time.strftime("%d_%m_%y_%H%M%S")+'.avi', fourcc, 20.0, (640, 480))
    #out1=cv2.VideoWriter("GRABACIONES/"+"contorno_"+time.strftime("%d_%m_%y_%H%M%S")+'.avi', fourcc, 20.0, (640, 480))
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        filename=("GRABACIONES/"+"Indicencias_"+time.strftime("%d_%m_%y_%H%M%S")+'.log'),
                        filemode='w', )
    logging.debug('REGISTRO DE INCIDENCIAS.')
    instanteInicial = time.time()

    instanteFinal = time.time()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, frame = cam.read()
                contornosimg=frame
                if ret == True:
                    movimiento=False
                    frame = cv2.resize(frame, (640, 480))
                    contornosimg = cv2.resize(contornosimg, (640, 480))
                    # IMPORTADO TUTORIAL MOVIMIENTO
                    # Aplicamos el algoritmo
                    fgmask = fgbg.apply(frame)

                    # Copiamos el umbral para detectar los contornos
                    contornosimg = fgmask.copy()

                    # Buscamos contorno en la imagen
                    im, contornos, hierarchy = cv2.findContours(contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # Recorremos todos los contornos encontrados
                    for c in contornos:
                        # Eliminamos los contornos más pequeños

                        if cv2.contourArea(c) < 500:
                            continue
                        else:
                            movimiento=True

                        # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
                        (x, y, w, h) = cv2.boundingRect(c)
                        # Dibujamos el rectángulo del bounds
                        cv2.rectangle(contornosimg, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    #image_np = cv2.resize(image_np, (640, 480))


                    print(movimiento)

                    if count > 3:
                        if ret == True:
                            # frame = cv2.flip(frame,0)
                            contadorMovimiento = 1
                            timestamp = datetime.datetime.now()
                            ts = timestamp.strftime("%d %B %Y %I:%M:%S%p")
                            cv2.rectangle(frame, (2, 220), (185, 235), (0, 0, 0), -1)
                            cv2.putText(frame,ts, (5, 230), cv2.FONT_ITALIC, 0.35, (255, 111, 255))
                            detectarObjetos(frame, detection_graph, category_index,sess)
                            print("grabando")
                            for x in range(20):
                                out.write(frame)
                            #out1.write(contornosimg)
                            #cv2.imshow("ventana", frame)
                    #else:
                        #movimiento=False


                    if movimiento == True:

                        count += 1
                        print("mov")
                        instanteInicial = time.time()

                        if count % 15 == 0:
                            print("movimiento a la hora " + time.strftime("%y_%m_%d_%H%M%S"))
                            nombre = "namepattern-" + time.strftime("%y-%m-%d- %H:%M:%S") + ".jpg"


                    else:
                        count = 0

                        instanteFinal = time.time()
                        tiempo = instanteFinal - instanteInicial  # Devuelve un objeto timedelta

                        print(tiempo)
                        if(tiempo>2):
                            #cuando vuelva a haber movimiento empezara un nuevo video
                            #out = cv2.VideoWriter("GRABACIONES/"+time.strftime("%d_%m_%y_%H%M%S") + '.avi', fourcc, 20.0, (640, 480))
                            instanteInicial=instanteFinal
                            contadorMovimiento = 0
                        else:
                            if(contadorMovimiento==1):
                                print("grabando")
                                #detectarObjetos(frame, detection_graph, category_index,sess)
                                timestamp = datetime.datetime.now()
                                ts = timestamp.strftime("%d %B %Y %I:%M:%S%p")
                                cv2.rectangle(frame, (2, 220), (185, 235), (0, 0, 0), -1)
                                cv2.putText(frame, ts, (5, 230), cv2.FONT_ITALIC, 0.35, (255, 111, 255))

                                out.write(frame)
                                #out1.write(contornosimg)
                                #cv2.imshow("ventana",frame)


                    movimiento = False


    print("camara")


def detectarObjetos(frame,detection_graph,category_index,sess):

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(frame, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=6)

    #cv2.imshow('Camara', frame)
    indices = []
    n = np.squeeze(classes).astype(np.int32)

    for inc in n:
        # print(inc)
        if (np.squeeze(scores)[inc] < 0.51):
            # preparar un algoritmo para la suma de los pesos menores a 0.5
            # print(np.squeeze(scores)[inc], category_index[n[inc]]['name'])
            indices.insert(len(indices), category_index[n[inc]]['name'])
            # print(len(indices))


    # ahora programar las sumas de cada elemento de la lista
    for inc in range(len(indices)):
        if (np.squeeze(scores)[inc] > 0.39):
            if(np.squeeze(scores)[inc] < 0.65):
                logging.info("Podría haber un %s con probabiidad de %s",str(category_index[n[inc]]['name']),str(round(np.squeeze(scores)[inc]*100)))
                #print(str(round(np.squeeze(scores)[inc]*100)))
            else:
                logging.warning("Atención, es probable que haya un %s al %s porcentaje de acierto", str(category_index[n[inc]]['name']),
                             str(round(np.squeeze(scores)[inc]*100)))
        #else:

            #print("Quizas haya ", category_index[n[inc]]['name'])


    print("object detection")



#####################
def grabar():
    print("grabar")

#main

camara()