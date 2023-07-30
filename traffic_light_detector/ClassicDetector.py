import cv2
import cv2
import numpy as np

class ClassicDetector:
  def __init__(self, img=None, tuning=False):    
    self._th1 = 96
    self._th2 = 104
    self._th3 = 164
    self._th4 = 96
    self._th5 = 91
    self._th6 = 116
    self._th7 = 89
    self._th8 = 20
    self._th9 = 40
    self._th10 = 171
    
    self._colors = [
      (0,0,255),
      (0,255,255),
      (0,255,0)
    ]
    
    self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    self._blur_size = 7
    
    self._img = img
    if tuning and img is None:
      raise ValueError('While tuning you also have to set image')
    
    
    self._tuning = tuning
    if self._tuning:
      self._window_name = 'test'
      cv2.namedWindow(self._window_name)
      cv2.createTrackbar('th1',  self._window_name, 0, 255, self._th1_cb)
      cv2.createTrackbar('th2',  self._window_name, 0, 255, self._th2_cb)
      cv2.createTrackbar('th3',  self._window_name, 0, 255, self._th3_cb)
      cv2.createTrackbar('th4',  self._window_name, 0, 255, self._th4_cb)
      cv2.createTrackbar('th5',  self._window_name, 0, 255, self._th5_cb)
      cv2.createTrackbar('th6',  self._window_name, 0, 255, self._th6_cb)
      cv2.createTrackbar('th7',  self._window_name, 0, 255, self._th7_cb)
      cv2.createTrackbar('th8',  self._window_name, 0, 255, self._th8_cb)
      cv2.createTrackbar('th9',  self._window_name, 0, 255, self._th9_cb)
      cv2.createTrackbar('th10',  self._window_name, 0, 255, self._th10_cb)
    
  def _th1_cb(self, val):
    self._th1 = val
    self.detect()
    
  def _th2_cb(self, val):
    self._th2 = val
    self.detect()
    
  def _th3_cb(self, val):
    self._th3 = val
    self.detect()
    
  def _th4_cb(self, val):
    self._th4 = val
    self.detect()
    
  def _th5_cb(self, val):
    self._th5 = val
    self.detect()
    
  def _th6_cb(self, val):
    self._th6 = val
    self.detect()
    
  def _th7_cb(self, val):
    self._th7 = val
    self.detect()
    
  def _th8_cb(self, val):
    self._th8 = val
    self.detect()
    
  def _th9_cb(self, val):
    self._th9 = val
    self.detect()
    
  def _th10_cb(self, val):
    self._th10 = val
    self.detect()
    
  def _mark_luminaire(self, stats, pos, img):
    shape = img.shape
    stats = stats.astype(float)
    W = stats[:,cv2.CC_STAT_WIDTH]
    H = stats[:,cv2.CC_STAT_HEIGHT]
    L = stats[:,cv2.CC_STAT_LEFT] - W*(2/36 + 4/36)
    T = stats[:,cv2.CC_STAT_TOP] - H* (2/36 + 8/36 + pos*42/36)
    
    W = W*(40 + 2*4)/36
    H = H*(124 + 20)/36
    
    out = np.zeros((stats.shape[0],4), float)
    out[:,0] = (L > 0)* L
    out[:,1] = (T > 0)* T
    out[:,2] = np.where(L < (shape[1]-W), W, shape[1] - L)
    out[:,3] = np.where(T < (shape[0]-H), H, shape[0] - T)
    
    return np.floor(out).astype(np.int32)
    
  def _threshold(self, S, Cb, Cr, th1, th2, th3,
                 s_type=cv2.THRESH_BINARY,
                 cr_type=cv2.THRESH_BINARY,
                 cb_type=cv2.THRESH_BINARY):
    _, I_1 = cv2.threshold(S,  th1, 255, s_type)
    _, I_2 = cv2.threshold(Cr, th2, 255, cr_type)
    _, I_3 = cv2.threshold(Cb, th3, 255, cb_type)
    
    I_12 = cv2.bitwise_and(I_1, I_2)
    I = cv2.bitwise_and(I_12, I_3)
    I = cv2.medianBlur(I, self._blur_size)
    I = cv2.morphologyEx(
      I, cv2.MORPH_OPEN,
      self._close_kernel
    )
    
    return I, I_1, I_2, I_3
  
  def _find_and_mark_circle(self, channel):
    total_area = channel.shape[0] * channel.shape[1]
    _, _, stats, centroids = cv2.connectedComponentsWithStats(~channel, 4, cv2.CV_32S)

    
    stats = np.delete(stats, np.any(np.isnan(centroids), axis=1), 0)
    stats_list = []
    
    for i in range(stats.shape[0]):
      x, y, width, height, area = stats[i]
      if 4*4 < width*height < 50*50:
        if width > 0 and height > 0:
          ratio = width / height
          if 0.6 < ratio < 1.4:
            stats_list.append(np.array(stats[i]))
    return np.array(stats_list)
  
  def detect(self, img=None, carla=False):
    if img is None:
      img = np.copy(self._img)
    else:
      img = np.copy(img)
      # if carla:
      #   cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      
    y, x, _ = img.shape
    roi_min_h = 1
    roi_max_h = y//5*4
    roi_min_w = x//5
    roi_max_w = x//5*4
    roi = img[roi_min_h:roi_max_h,roi_min_w:roi_max_w,:]
      
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]
    Cr = ycrcb[:,:,1]
    Cb = ycrcb[:,:,2]
    
    R, R1, R2, R3 = self._threshold(S, Cr, Cb, self._th1, self._th2, self._th3,
                 s_type=cv2.THRESH_BINARY,
                 cr_type=cv2.THRESH_BINARY_INV,
                 cb_type=cv2.THRESH_BINARY)
    G, G1, G2, G3 = self._threshold(S, Cr, Cb, self._th4, self._th5, self._th6,
                 s_type=cv2.THRESH_BINARY,
                 cr_type=cv2.THRESH_BINARY_INV,
                 cb_type=cv2.THRESH_BINARY_INV)
    Y, Y1, Y2, Y3 = self._threshold(S, H, H, self._th7, self._th8, self._th9,
                 s_type=cv2.THRESH_BINARY,
                 cr_type=cv2.THRESH_BINARY,
                 cb_type=cv2.THRESH_BINARY_INV)
  
    _, Y4 = cv2.threshold(V, self._th10, 255, cv2.THRESH_BINARY)
    Y = cv2.bitwise_and(Y, Y4)
    
    r_stats = self._find_and_mark_circle(R)
    g_stats = self._find_and_mark_circle(G)
    y_stats = self._find_and_mark_circle(Y)
    
    stats_arr = [r_stats, y_stats, g_stats]
    
    bboxes = None
    colors = None
    
    for i in range(3):
      if stats_arr[i].shape[0] > 0:
        if bboxes is None:
          bboxes = self._mark_luminaire(stats_arr[i], i, img)
          colors = [i]
        else:
          bboxes = np.append(bboxes, self._mark_luminaire(stats_arr[i], i, img), axis=0)
          colors.append(i)

    if bboxes is None:
      bboxes = np.array([[]])
      colors = np.array([[]])
    bboxes = bboxes.reshape((-1,4))
    
    for bbox, color in zip(bboxes, colors):
      bbox[0] += roi_min_w
      bbox[1] += roi_min_h
      x_min = bbox[0]
      x_max = x_min + bbox[2]
      y_min = bbox[1]
      y_max = y_min + bbox[3]
      cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), self._colors[color], 1)
      cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), self._colors[color], 1)
      cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), self._colors[color], 1)
      cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), self._colors[color], 1)
      
    
    if self._tuning:
      R = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
      G = cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)
      Y = cv2.cvtColor(Y, cv2.COLOR_GRAY2BGR)
      
      R1 = cv2.cvtColor(R1, cv2.COLOR_GRAY2BGR)
      R2 = cv2.cvtColor(R2, cv2.COLOR_GRAY2BGR)
      R3 = cv2.cvtColor(R3, cv2.COLOR_GRAY2BGR)
      
      G1 = cv2.cvtColor(G1, cv2.COLOR_GRAY2BGR)
      G2 = cv2.cvtColor(G2, cv2.COLOR_GRAY2BGR)
      G3 = cv2.cvtColor(G3, cv2.COLOR_GRAY2BGR)
      
      Y1 = cv2.cvtColor(Y1, cv2.COLOR_GRAY2BGR)
      Y2 = cv2.cvtColor(Y2, cv2.COLOR_GRAY2BGR)
      Y3 = cv2.cvtColor(Y3, cv2.COLOR_GRAY2BGR)
      Y4 = cv2.cvtColor(Y4, cv2.COLOR_GRAY2BGR)
      
      roi = img[roi_min_h:roi_max_h,roi_min_w:roi_max_w,:]
      
      # out = np.concatenate((R, G, Y, roi), axis=1)
      # out = np.concatenate((R1, R2, R3, R, roi), axis=1)
      # out = np.concatenate((G1, G2, G3, G, roi), axis=1)
      out = np.concatenate((Y1, Y2, Y3, Y4, Y, roi), axis=1)
      w, h, _ = out.shape
      out = cv2.resize(out, (h//2,w//2), interpolation = cv2.INTER_AREA)
      cv2.imshow(self._window_name, out)

    # if carla:
    #   cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return bboxes, colors, img

if __name__ == '__main__':
  img = cv2.imread('images/000211.png')
  
  tuning = True
  
  detector = ClassicDetector(img, tuning)
  if not tuning:
    sign = detector.detect(img=img)
    cv2.imshow('result', sign)
  
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  
  
  