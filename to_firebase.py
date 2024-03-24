import pyrebase
import gzip
import json
from threading import Thread, Lock
import time
import cv2

class to_firebase():
    def __init__(self, img_name):

      config = {
          "apiKey": "AIzaSyBpbS4rEI73PAOQ8jH5R11ZrC4wF9vOC5g",
          "authDomain": "remotecameras-c976b.firebaseapp.com",
          "projectId": "remotecameras-c976b",
          "storageBucket": "remotecameras-c976b.appspot.com",
          "messagingSenderId": "1070375402923",
          "appId": "1:1070375402923:web:57d650eba9a31e5e6726ee",
          "measurementId": "G-5KGJ6PCF4B",
          "serviceaccount": "serviceAccount.json",
          "databaseURL": "https://remotecameras-c976b-default-rtdb.firebaseio.com/",
          "serviceAccount": "serviceAccount.json"
      }


      firebase = pyrebase.initialize_app(config)
      self.storage = firebase.storage()
      self.img_name = img_name

    def start(self, img):
        self.thread = Thread(target=self.update, args=(img,))
        self.thread.start()
        return self
        
    def update(self, img):
        tic = time.time()
        cv2.imwrite("tmp.jpg",img)
        print(self.img_name)
        self.storage.child('/lagunas/'+self.img_name).put("tmp.jpg")
        print(time.time() - tic)
        

