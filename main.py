# 3
# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from tkinter import *
from imutils.video import VideoStream
from imutils.video import FPS
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from sklearn.svm import SVC
import PIL.Image, PIL.ImageTk
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os



def extract_embeddings(detector, embedder):

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images("dataset"))	

    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    #TODO: posible: solo procesr imagenes pendientes
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]	

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]	

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

    	# apply OpenCV's deep learning-based face detector to localize
    	# faces in the input image
    	detector.setInput(imageBlob)
    	detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > 0.5: #confidence
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("output/embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()

    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open("output/embeddings.pickle", "rb").read())

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open("output/recognizer.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open("output/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()

    return knownNames
    #def createNewDataset(frame, name):
    #TODO: crear un nuevo dataset con labels incluido
    #Mirar si ya existe (ver si la foto encaja con alguna ya dada)
    #si no 
    #Crear una carpeta con nombre dado
    #guardar los fotogramas necesarios (mirar en la web)
    #volver a generar dl
    #anyadirlo a la lista

def appInit(window, window_title, canvas, nombres):
    window.title(window_title)
    canvas.pack()

    btn= Button(window, text='Nuevo usuario', command=extract_embeddings)
    btn.pack(anchor=CENTER, expand=True)

    listbox = Listbox(window)
    listbox.place(x = 0, y = 450)
    listbox.insert(0, *nombres)    

    window.update()

def updateFrame(window,frame,canvas):
    #para que salga en el color que toca
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0, 0, image = photo, anchor = NW)
    window.update()

if __name__ == "__main__":
    
    
    print("[INFO] loading face detector...")
    # load our serialized face detector from disk
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath =  "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    # load our serialized face detector from disk
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    
    print("[INFO] loading face recognizer...")
    # load our serialized face embedding model from disk
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    #creamos el modelo al principio y guardamos los nombres para la lista
    knownNames = extract_embeddings(detector, embedder)
    names = list(set(knownNames))
    names.remove('unknown')
    print(names)

    #interface
    interface = Tk()
    canvas = Canvas(interface, width = 600, height = 600)
    appInit(interface, "Reconocimiento facial OpenCV", canvas, names)

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())

    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # start the FPS throughput estimator
    fps = FPS().start()

    # loop over frames from the video file stream
    while True:
        
        # grab the frame from the threaded video stream
        frame = vs.read()

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.7:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # update the FPS counter
        fps.update()

        #para la parte de interfaz
        updateFrame(interface,frame, canvas)
    
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()