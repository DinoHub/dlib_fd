import os
import cv2
import dlib
import numpy as np

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def cvbb2dlibrect(bb):
    x,y,w,h = bb
    return dlib.rectangle(left=x,top=y,right=x+w,bottom=y+h)

def cvbbs2bbs(cv_bbs):
    '''
    Converts cv's bbs to bbs
    '''
    bbs = []
    for cv_bb in cv_bbs:
        x,y,w,h = cv_bb
        bb = {'rect':{'t': y, 
                      'l': x, 
                      'r': x+w, 
                      'b': y+h, 
                      'w': w, 
                      'h': h},
              # 'confidence': dlib_rect.confidence}
              'confidence': 1.0} #haar does not give confidence, so just assume 1.0
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


class cv2haar_FD:
    def __init__(self, fd_xml=None, landmarks_dat=None, scaleFactor=1.1, minNeighbors=8, max_n=None, upsampling=None):
        if fd_xml is None:
            XML= os.path.join(CURR_DIR,'haarcascade_frontalface_default.xml')
            assert os.path.exists(XML)
        if landmarks_dat is None:
            landmarks_dat = os.path.join(CURR_DIR, 'shape_predictor_68_face_landmarks.dat')
            assert os.path.exists(landmarks_dat),'{} does not exists'.format(landmarks_dat)

        self.face_detector = cv2.CascadeClassifier(XML)
        self.predictor = dlib.shape_predictor(landmarks_dat)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.max_n = max_n
        # self.i = 0
        print('cv2 haar cascade face detector initialised!')

    def detect(self, frame):
        bbs = self.face_detector.detectMultiScale(frame, 
                                            scaleFactor = self.scaleFactor, 
                                            minNeighbors = self.minNeighbors)
        if isinstance(bbs, tuple):
            return []
        else:
            return bbs
    
    def _detect_batch(self, img3chnls):
        '''
        :return: array of modd_rect, each containing confidence & rect in [(l,t),(r,b)] format.
        '''
        assert img3chnls is not None,'FD didnt rcv img'
        all_bbs = []
        for img3chnl in img3chnls:
            try:
                if img3chnl is None or img3chnl.dtype != np.uint8:
                    all_bbs.append([])
                else:
                    all_bbs.append(self.detect(img3chnl))
            except Exception as e:
                print("WARNING from FD detect_batch: {}".format(e))
                all_bbs.append([])
        # print(all_bbs)
        return all_bbs

    def detect_align_faces_batch(self, img3chnls, imgDim=96, num_face=None):
        all_cvbbs = self._detect_batch(img3chnls)
        all_aligned_faces = []
        all_bbs = []
        for i, bbs in enumerate(all_cvbbs):
            if len(bbs)==0:
                aligned_faces = []                
            else:
                if self.max_n is not None:
                    bbs = sorted(bbs, key=lambda bb: bb[2]*bb[3],# width*height
                                 reverse=True)[:self.max_n]
                aligned_faces = self._align_batch(img3chnls[i], bbs, imgDim)
                # aligned_faces = [self._align_one_68(img3chnls[i], mmod_bb.rect, imgDim, INNER_EYES_AND_BOTTOM_LIP) for mmod_bb in bbs]


            all_bbs.append(cvbbs2bbs(bbs))
            all_aligned_faces.append(aligned_faces)
        # print(all_bbs)
        return all_bbs, all_aligned_faces

    def _align(self,img3chnl, bb, imgDim):
        bb_rect = cvbb2dlibrect(bb)
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
            # self.i+=1
        return aligned_faces


if __name__=='__main__':
    test = '/home/dh/Workspace/FR/fronthedge/outFrames/Webcam0_1.png'
    img = cv2.imread(test)
    print(img.shape)
    bbs = detect(img)
    print(bbs)