# from https://github.com/IrvingMeng/MagFace faces should be aligned to 112x112 with 5 landmarks, and save a .list file with image information
# this project uses retinaface to align the faces

from retinaface import RetinaFace


def getlandmarks(face_image):
    "Function to get landmarks"
    resp = RetinaFace.detect_faces(face_image)
    landmarks = resp[0]["landmarks"]
    
    