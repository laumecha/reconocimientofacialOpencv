# import the necessary packages
from tkinter import *
from imutils.video import VideoStream
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from ctypes import *
from sklearn.svm import SVC
import PIL.Image, PIL.ImageTk
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import termios

global detector
global embedder
global knownNames
global names
global recognizer
global le
global datapath

def extract_embeddings():

    # guardamos los paths de las personas guardadas
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images("users"))	

    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownEmbeddings = [] 
    global knownNames
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
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
    embeddings_path = os.path.join(datapath,"output", "embeddings.pickle")
    f = open(embeddings_path, "wb")
    f.write(pickle.dumps(data))
    f.close()

    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())

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
    recognizer_path = os.path.join(datapath,"output", "recognizer.pickle")
    f = open(recognizer_path, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    le_path= os.path.join(datapath,"output", "le.pickle")
    f = open(le_path, "wb")
    f.write(pickle.dumps(le))
    f.close()

def reload():
    global datapath
    datapath = 'data'
    print("[INFO] loading face detector...")
    #cargamos la configuracion de las capas de la red neuronal
    protoPath = os.path.join(datapath,"face_detection_model", "deploy.prototxt")
    #cargamos el modelo de Caffe para reconocer caras
    modelPath = os.path.join(datapath,"face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel")
    
    # load our serialized face detector from disk
    global detector
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    
    print("[INFO] loading face recognizer...")
    # load our serialized face embedding model from disk
    global embedder
    embedder_path = os.path.join(datapath, "openface_nn4.small2.v1.t7")
    embedder = cv2.dnn.readNetFromTorch(embedder_path)

    #guardamos los nombres para la lista y borramos duplicados
    extract_embeddings()
    global names
    names = list(set(knownNames))
    names.remove('unknown')

    global recognizer
    global le

    # load the actual face recognition model along with the label encoder
    recognizer_path = os.path.join(datapath,"output", "recognizer.pickle")
    recognizer = pickle.loads(open(recognizer_path, "rb").read())
    le_path = os.path.join(datapath,"output", "le.pickle")
    le = pickle.loads(open(le_path, "rb").read())

def appInit(window_title):

    interface.title("Reconocimiento facial OpenCV")
    canvas.pack()

    btn = Button(interface, text='Nuevo usuario', command=newUser)
    btn.place(x = 200, y = 500)

    listbox.place(x = 0, y = 450)
    listbox.insert(0, *names)    

    interface.update()

def updateFrame():
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    canvas.create_image(0, 0, image = photo, anchor = NW)
    interface.update()

def newUser():
    newWindow = Toplevel(interface)
    newWindow.title("Nuevo usuario")
    newWindow.geometry("350x300")

    text = Entry(newWindow)
    text.place(x=50, y=50)

    labelText.set("Pulsa el boton para empezar")    
    label = Label( newWindow, textvariable=labelText, relief=RAISED )
    label.pack()

    btn= Button(newWindow, text='Aceptar', command= lambda: acceptButtonNewUser(text))
    btn.pack(anchor=CENTER, expand=True)

def acceptButtonNewUser(text):
    name = text.get()
    if len(name) != 0:
        done = makeDir(name)
        if done:
            for x in range(0, 5):
                updateFrame()
                captura = vs.read()
                message = " Faltan " + str(5-(x+1)) + " capturas"
                labelText.set(message)  
                time.sleep(1)
                saveCaptura(name,x)
            labelText.set("Pulsa el boton para empezar")  
            listbox.insert(END, name)
            reload()
        else :
            message = "Introduce un nombre no existente"
            labelText.set(message) 
    else :
        message = "Introduce un nombre"
        labelText.set(message) 

def makeDir(dirName):
    res = isinstance(dirName, str) 

    if res: #esto no funciona
        try:
            # Create target Directory
            os.mkdir('users/'+dirName)
            print("Directory " , dirName ,  " Created ") 
            return True
        except: #esto no sabe que es
            print("Directory " , dirName ,  " already exists")
            return False
    else:
        print("ckech the input")
        return False
    
def saveCaptura(name,x):
    output = name + str(x) + ".jpg"
    save_path = os.path.join('users', name, output)
    cv2.imwrite(save_path, captura)


if __name__ == "__main__":   

    reload()

    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    #interfaz
    interface = Tk()
    canvas = Canvas(interface, width = 600, height = 600) 
    listbox = Listbox(interface)
    labelText = StringVar()
    appInit( "Reconocimiento facial OpenCV")

    frame = vs.read()
    captura = frame

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()
        captura = frame

        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        #para manteren la imagen sin el cuadro

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

        #para la parte de interfaz
        updateFrame()

    cv2.destroyAllWindows()
    vs.stop()