#!/usr/bin/python3
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to run a CNN based face detector using dlib.  The
#   example loads a pretrained model and uses it to find faces in images.  The
#   CNN model is much more accurate than the HOG based model shown in the
#   face_detector.py example, but takes much more computational power to
#   run, and is meant to be executed on a GPU to attain reasonable speed.
#
#   You can download the pre-trained model from:
#       http://dlib.net/files/mmod_human_face_detector.dat.bz2
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg
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
#       python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
#   if you have a CPU that supports AVX instructions, you have an Nvidia GPU
#   and you have CUDA installed since this makes things run *much* faster.
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

def dlibRects2bbs(dlib_rects):
    '''
    Converts dlib's dlib_rects to bbs
    '''
    bbs = []
    for dlib_rect in dlib_rects:
        bb = {'rect':{'t':dlib_rect.top(), 
                      'l':dlib_rect.left(),
                      'r':dlib_rect.right(),
                      'b':dlib_rect.bottom(), 
                      'w':dlib_rect.width(), 
                      'h':dlib_rect.height()},
              # 'confidence': dlib_rect.confidence}
              'confidence': 1.0} #HOG does not give confidence, so just assume 1.0
        bbs.append(bb)
    return bbs

TEMPLATE = np.float32([
        (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
        (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
        (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
        (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
        (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
        (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
        (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
        (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
        (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
        (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
        (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
        (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
        (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
        (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
        (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
        (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
        (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
        (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
        (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
        (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
        (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
        (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
        (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
        (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
        (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
        (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
        (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
        (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
        (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
        (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
        (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
        (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
        (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
        (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

#: Landmark indices.
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]

class hog_FD:
    def __init__(self, landmarks_dat=None, upsampling=0, max_n =None ):
        if landmarks_dat is None:
            landmarks_dat = os.path.join(CURR_DIR, 'shape_predictor_68_face_landmarks.dat')
            # landmarks_dat = os.path.join(CURR_DIR, 'shape_predictor_5_face_landmarks.dat')
            assert os.path.exists(landmarks_dat),'{} does not exists'.format(landmarks_dat)

        # assert fd_dat is not None,'cnn fd model not given'
        # assert landmarks_dat is not None,'landmarks_dat not given'

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmarks_dat)
        self.detector_upsampling = upsampling
        self.max_n = max_n
        # warm up
        # self.detector(np.zeros((10,10,3), dtype=np.uint8),0)
        self.i = 0
        print("FACE DETECTION: dlib's HOG FD object initialised!")

    def detect(self, img3chnl):
        '''
        :return: array of modd_rect, each containing confidence & rect in [(l,t),(r,b)] format.
        '''
        assert img3chnl is not None,'FD didnt rcv img'
        
        try:
            mmod_bbs = self.detector(img3chnl, self.detector_upsampling)
            # mmod_bbs are array of mmod_bb, each contains confidence & rect
            return mmod_bbs
        except Exception as e:
            print("WARNING from FD detect: {}".format(e))
            # import inspect
            # for i in inspect.getmembers( self.detector ):
            #     print(i)
            # print( dir(self.detector) )
            return []

    def _detect_batch(self, img3chnls):
        '''
        :return: array of modd_rect, each containing confidence & rect in [(l,t),(r,b)] format.
        '''
        assert img3chnls is not None,'FD didnt rcv img'
        # all_mmod_bbs = []
        all_dlib_rects = []
        for img3chnl in img3chnls:
            try:
                if img3chnl is None or img3chnl.dtype != np.uint8:
                    all_dlib_rects.append([])
                else:
                    all_dlib_rects.append(self.detector(img3chnl, self.detector_upsampling))
                    # print(img3chnl.shape)
                    # print(img3chnl.dtype)
                # mmod_bbs are array of mmod_bb, each contains confidence & rect
            except Exception as e:
                print("WARNING from FD detect_batch: {}".format(e))
                # print(type(img3chnl))
                # print(img3chnl.shape)
                # print(img3chnl[1][1])
                # print(img3chnl.dtype)
                all_dlib_rects.append([])
        return all_dlib_rects

    def _align(self,img3chnl, bb_rect, imgDim):
        # aligned_face = self.aligner.align(imgDim, img3chnl, bb = bb_rect)
        points = self.predictor(img3chnl, bb_rect)
        landmarks = list(map(lambda p:(p.x, p.y), points.parts()))
        npLandmarks = np.float32(landmarks)
        npLandmarksIndices = np.array(INNER_EYES_AND_BOTTOM_LIP) #landmark indices = INNER_EYES_AND_BOTTOM_LIP
        H = cv2.getAffineTransform(npLandmarks[npLandmarksIndices],
                                   imgDim * MINMAX_TEMPLATE[npLandmarksIndices])
        aligned_face = cv2.warpAffine(img3chnl, H, (imgDim, imgDim))
        return aligned_face

    def _align_batch(self, img3chnl, bbs, imgDim):
        '''
        For batch faces
        '''
        assert img3chnl is not None, 'Landmark predictor didnt rcv img'
        assert bbs is not None, 'Landmark predictor didnt rcv bb'
        aligned_faces = []
        for bb in bbs:
            aligned_face = self._align(img3chnl, bb, imgDim)
            # cv2.imwrite('1/aligned_{}.jpg'.format(self.i), aligned_face)
            # cv2.imwrite('img.jpg', img3chnl)
            aligned_faces.append(aligned_face)
            self.i+=1
        return aligned_faces

    def detect_align_faces(self, img3chnl, imgDim=96, num_face=None):
        all_bbs, all_aligned_faces = self.detect_align_faces_batch([img3chnl])
        return all_bbs[0], all_aligned_faces[0]

    def detect_align_faces_batch(self, img3chnls, imgDim=96, num_face=None):
        all_dlib_rects = self._detect_batch(img3chnls)
        all_aligned_faces = []
        all_bbs = []
        for i, dlib_rects in enumerate(all_dlib_rects):
            if len(dlib_rects)==0:
                aligned_faces = []                
            else:
                if self.max_n is not None:
                    dlib_rects = sorted(dlib_rects, key=lambda dlib_rect: dlib_rect.width() * dlib_rect.height(), reverse=True)[:self.max_n]
                aligned_faces = self._align_batch(img3chnls[i], dlib_rects, imgDim)

            # all_bbs.append(mmod2bbs(dlib_rects))
            all_bbs.append(dlibRects2bbs(dlib_rects))
            all_aligned_faces.append(aligned_faces)
        return all_bbs, all_aligned_faces

    def _align_one_68(self, img3chnl, mmod_bb_rect, imgDim, landmarkIndices):
        '''
        For one face
        '''
        assert img3chnl is not None, 'Landmark predictor didnt rcv img'
        assert mmod_bb_rect is not None, 'Landmark predictor didnt bb'
        points = self.predictor(img3chnl, mmod_bb_rect)
        landmarks = list(map(lambda p:(p.x, p.y), points.parts()))
        print(landmarks)
        npLandmarks = np.float32(landmarks)
        npLandmarksIndices = np.array(landmarkIndices)
        H = cv2.getAffineTransform(npLandmarks[npLandmarksIndices],
                                   imgDim * MINMAX_TEMPLATE[npLandmarksIndices])
        aligned_face = cv2.warpAffine(img3chnl, H, (imgDim, imgDim))
        cv2.imwrite('aligned.jpg', aligned_face)
        cv2.imwrite('img.jpg', img3chnl)
        return aligned_face

##########


if __name__ == "__main__":
    import time
    faceDet = cnn_FD(upsampling=0, max_n = 5)
    image = cv2.imread('/home/dh/Workspace/FR/master_fr/dlib_FD/test_frame1.png')
    # mmod_bbs = faceDet._detect(image) # warm up
    # print('initial size:{}'.format(image.shape))
    
    # big_frame = np.hstack ((image, image))
    # big_frame = np.vstack ((big_frame, big_frame))

    #stitch
    # k = 4
    # big_frame = None
    # for _ in range(k):
    #     if big_frame is None:
    #         big_frame = image
    #     else:
    #         big_frame = np.hstack ((big_frame, image))
    # print('big size:{}'.format(big_frame.shape))
    k = 2
    frames = [image] * k

    n = 10
    start = time.time()

    for _ in range(n):
        bbs, aligned_faces = faceDet.detect_align_faces_batch(frames)

    print('time for detect: {}, {}'.format((time.time()-start)/n, bbs))

    # from skimage import io
    # import sys
    # import time

    # if len(sys.argv) < 3:
    #     print(
    #         "Call this program like this:\n"
    #         "   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
    #         "You can get the mmod_human_face_detector.dat file from:\n"
    #         "    http://dlib.net/files/mmod_human_face_detector.dat.bz2")
    #     exit()

    # cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.argv[1])
    # win = dlib.image_window()

    # for f in sys.argv[2:]:
    #     print("Processing file: {}".format(f))
    #     img = []
    #     batch_size = 16
    #     for i in range(batch_size):
    #         img.append(io.imread(f))
    #     # img = io.imread(f)
    #     # The 1 in the second argument indicates that we should upsample the image
    #     # 1 time.  This will make everything bigger and allow us to detect more
    #     # faces.
    #     n = 100
    #     dur = 0.0
    #     for i in range(n):
    #         start = time.time()
    #         dets = cnn_face_detector(img, 0, batch_size = batch_size)
    #         # dets = cnn_face_detector(img, 0)
    #         this_dur = time.time() - start
    #         dur +=  this_dur
    #         print(this_dur)
    #         '''
    #         This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
    #         These objects can be accessed by simply iterating over the mmod_rectangles object
    #         The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.
            
    #         It is also possible to pass a list of images to the detector.
    #             - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

    #         In this case it will return a mmod_rectangless object.
    #         This object behaves just like a list of lists and can be iterated over.
    #         '''
    #         # print("Number of faces detected: {}".format(len(dets)))
    #         for i, d in enumerate(dets):
    #             print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
    #                 i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

    #         rects = dlib.rectangles()
    #         rects.extend([d.rect for d in dets])
    #     print('Rate: ',dur/n)

    #     win.clear_overlay()
    #     win.set_image(img)
    #     win.add_overlay(rects)
    #     dlib.hit_enter_to_continue()