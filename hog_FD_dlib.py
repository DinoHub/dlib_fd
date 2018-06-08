#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image.  In
#   particular, it shows how you can take a list of images from the command
#   line and display each on the screen with red boxes overlaid on each human
#   face.
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./face_detector.py ../examples/faces/*.jpg
#
#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.  This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.  Therefore, if you are interested in making
#   your own object detectors then read the train_object_detector.py example
#   program.  
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 
import dlib
import os 
import numpy as np
import cv2


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def rect2bbs(rect_bbs):
    '''
    Converts dlib's mmod_bbs to bbs
    '''
    bbs = []
    for rect_bb in rect_bbs:
        bb = {'rect':{'t':rect_bb.top(), 
                      'l':rect_bb.left(),
                      'r':rect_bb.right(),
                      'b':rect_bb.bottom(), 
                      'w':rect_bb.width(), 
                      'h':rect_bb.height()},
              'confidence':0.99 }
        bbs.append(bb)
    return bbs

class hog_FD:
    def __init__(self, landmarks_dat=None, upsampling=0, max_n =None ):
        if landmarks_dat is None:
            landmarks_dat = os.path.join(CURR_DIR, 'shape_predictor_68_face_landmarks.dat')
            assert os.path.exists(landmarks_dat),'{} does not exists'.format(landmarks_dat)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmarks_dat)
        self.detector_upsampling = upsampling
        self.max_n = max_n
        print("FACE DETECTION: dlib's HOG FD object initialised!")

    def _detect(self, img3chnl):
        assert img3chnl is not None,'FB didnt rcv img'
        try:
            rect_bbs = self.detector(img3chnl, self.detector_upsampling)
            # mmod_bbs are array of mmod_bb, each contains confidence & rect
            return rect_bbs
        except Exception as e:
            print("WARNING from FD: {}".format(e))
            return []
    
    def _align(self, img3chnl, bbs, imgDim):
        '''
        For batch faces
        '''
        assert img3chnl is not None, 'Landmark predictor didnt rcv img'
        assert bbs is not None, 'Landmark predictor didnt rcv bb'
        faces = dlib.full_object_detections()
        for bb in bbs:
            faces.append(self.predictor(img3chnl, bb))
        aligned_faces = dlib.get_face_chips(img3chnl, faces, size=imgDim)
        return aligned_faces

    def detect_align_faces(self, img3chnl, imgDim=96, num_face=None):
        rect_bbs = self._detect(img3chnl)
        if rect_bbs is None or len(rect_bbs)==0:
            return [], []

        if self.max_n is not None:
            rect_bbs = sorted(rect_bbs, key=lambda rect: rect.width() * rect.height(), reverse=True)[:self.max_n]

        aligned_faces = self._align(img3chnl, rect_bbs, imgDim)
        # aligned_faces = [self._align(img3chnl, rect_bbs, imgDim, landmarkIndices) for rect_bb in rect_bbs]

        return rect2bbs(rect_bbs), aligned_faces
    



    #legacy 68 landmarks predictor
    def _align_one_68(self, img3chnl, mmod_bb_rect, imgDim, landmarkIndices):
        '''
        For one face
        '''
        assert img3chnl is not None, 'Landmark predictor didnt rcv img'
        assert mmod_bb_rect is not None, 'Landmark predictor didnt bb'
        points = self.predictor(img3chnl, mmod_bb_rect)
        landmarks = list(map(lambda p:(p.x, p.y), points.parts()))
        npLandmarks = np.float32(landmarks)
        npLandmarksIndices = np.array(landmarkIndices)
        H = cv2.getAffineTransform(npLandmarks[npLandmarksIndices],
                                   imgDim * MINMAX_TEMPLATE[npLandmarksIndices])
        aligned_face = cv2.warpAffine(img3chnl, H, (imgDim, imgDim))
        return aligned_face

# ######### Can be deleted. Only used for the old 68 landmark predictor.
# TEMPLATE = np.float32([
#     (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
#     (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
#     (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
#     (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
#     (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
#     (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
#     (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
#     (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
#     (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
#     (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
#     (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
#     (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
#     (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
#     (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
#     (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
#     (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
#     (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
#     (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
#     (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
#     (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
#     (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
#     (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
#     (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
#     (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
#     (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
#     (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
#     (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
#     (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
#     (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
#     (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
#     (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
#     (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
#     (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
#     (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

# TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
# MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

# #: Landmark indices.
# INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
# OUTER_EYES_AND_NOSE = [36, 45, 33]

if __name__ == "__main__":

    import sys
    from skimage import io
    import time

    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()

    for f in sys.argv[1:]:
        print("Processing file: {}".format(f))
        img = io.imread(f)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        n = 100
        dur = 0.0

        for i in range(n):
            start = time.time()
            dets = detector(img, 0)
            dur += time.time() - start
            print("Number of faces detected: {}".format(len(dets)))
            for i, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))
        print('FD rate:',dur/n)

        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        dlib.hit_enter_to_continue()


    # Finally, if you really want to you can ask the detector to tell you the score
    # for each detection.  The score is bigger for more confident detections.
    # The third argument to run is an optional adjustment to the detection threshold,
    # where a negative value will return more detections and a positive value fewer.
    # Also, the idx tells you which of the face sub-detectors matched.  This can be
    # used to broadly identify faces in different orientations.
    if (len(sys.argv[1:]) > 0):
        img = io.imread(sys.argv[1])
        dets, scores, idx = detector.run(img, 1, -1)
        for i, d in enumerate(dets):
            print("Detection {}, score: {}, face_type:{}".format(
                d, scores[i], idx[i]))