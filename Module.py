
def read_image():
    import cv2 as cv
    img=cv.imread('Photos/cat.jpg')
    cv.imshow('Photo',img)
    cv.waitKey(0)
  

def gray_image():
  import cv2 as cv
  img = cv.imread('Photos/park.jpg')
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
  cv.imshow('Gray',gray)


def HSV_image():
  import cv2 as cv
  img = cv.imread('Photos/park.jpg')
  hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
  cv.imshow('HSV', hsv)


def LAB_image():
  import cv2 as cv
  img = cv.imread('Photos/park.jpg')
  lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
  cv.imshow('LAB', lab)


def RGB_img():
  import cv2 as cv
  img = cv.imread('Photos/park.jpg')
  rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
  cv.imshow('RGB', rgb)


def HSV_to_BGR():
   import cv2 as cv
   img = cv.imread('Photos/park.jpg')
   hsv_bgr = cv.cvtColor('hsv', cv.COLOR_HSV2BGR)
   cv.imshow('HSV --> BGR', hsv_bgr)


def read_video():
  import cv2 as cv
  capture = cv.VideoCapture('Videos/dog.mp4')
  while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
  capture.release()
  cv.destroyAllWindows()


def resize():
  import cv2 as cv
  img=cv.imread('Photos/cat.jpg')
  cv.imshow('Cat',img)
  def rescaleFrame(frame,scale=0.75):
        width=int(frame.shape[1]*scale)
        height=int(frame.shape[0]*scale)
        dimensions=(width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
  resized_image=rescaleFrame(img)
  cv.imshow('Resized Cat',resized_image)
  cv.waitKey(0)


def rescale():
  import cv2 as cv
  def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
  capture = cv.VideoCapture('Videos/dog.mp4')
  while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)
    cv.imshow('Video', frame)
    cv.imshow('Resized_frame', frame_resized)
    if cv.waitKey(20) & 0xFF==ord('d'):
       break
  capture.release()
  cv.destroyAllWindows()


def shapes():
  import cv2 as cv
  import numpy as np
  blank = np.zeros((500,500,3), dtype='uint8')
  
  cv.imshow ('blank', blank)
  blank[200:300, 300:400] = 0,255,0
  cv.imshow('green', blank)

  cv.rectangle(blank, (0,0),(blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)
  cv.imshow('recrangle', blank)

  cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,255,0), thickness=3)
  cv.imshow('circle', blank)

  cv.line(blank,(0,0),(300,400),(255,255,255), thickness=3)
  cv.imshow('line', blank)

  cv.waitKey(0)


def text():
  import cv2 as cv
  import numpy as np
  blank=np.zeros((500,500,3),dtype='uint8')
  cv.putText(blank,'hello',(0,255),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),thickness=3)
  cv.imshow('Text',blank)
  cv.waitKey(0)


def blur_gaussian():
  import cv2 as cv
  img = cv.imread('Photos/park.jpg')
  blur = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
  cv.imshow('Gaussian' ,blur)


def dilate():
  import cv2 as cv
  img = cv.imread('Photos/park.jpg')
  canny = cv.Canny(img,125,175)
  dilated = cv.dilate(canny,(3,3),iterations=1)
  cv.imshow('Dilated',dilated)


def erode():
  import cv2 as cv
  img = cv.imread('Photos/park.jpg')
  eroded = cv.erode('dilated',(3,3),iterations=1)
  cv.imshow('Eroded',eroded)


def resize():
  import cv2 as cv
  img = cv.imread('Photos/park.jpg')
  resize = cv.resize(img, (500,500), interpolation=cv.INTER_AREA)
  cv.imshow('resized', resize)
  cv.waitKey(0)


def cropped():
   import cv2 as cv
   img = cv.imread('Photos/park.jpg')
   cropped = img[50:200,200:400]
   cv.imshow('Cropped',cropped)


def img_translation():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/park.jpg')
  def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

  translated = translate(img, 100,100)
  cv.imshow('Translated', translated)


def img_rotation():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/park.jpg')
  def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2, height//2)
        rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
        dimensions = (width,height)
        return cv.warpAffine(img, rotMat, dimensions)
  rotated = rotate(img,45)
  cv.imshow('Rotated', rotated)


def img_flip_vertical():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/park.jpg', 0 )
  flip = cv.flip(img,100)
  cv.imshow('Flip Vertically',flip)
  cv.waitKey(0)
  cv.destroyAllWindows()


def img_flip():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/park.jpg', 0 )
  flip = cv.flip(img,0)
  cv.imshow('Flip',flip)
  cv.waitKey(0)
  cv.destroyAllWindows()


def img_shrink():
     import cv2 as cv
     import numpy as np
     img = cv.imread('Photos/park.jpg', 0)
     rows, cols = img.shape
     img_shrinked = cv.resize(img, (250,200), interpolation=cv.INTER_AREA)
     cv.imshow('shrinked', img_shrinked)
     cv.waitKey(0)
     cv.destroyAllWindows()

def img_enlarge():
     import cv2 as cv
     import numpy as np
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     img_enlarged = cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
     cv.imshow('enlarged', img_enlarged)
     cv.waitKey(0)
     cv.destroyAllWindows()


def shearing_x_axis():
     import cv2 as cv
     import numpy as np
     img = cv.imread('Photos/park.jpg', 0)
     rows, cols = img.shape
     M = np.float32([[1,0.5,0], [0,1,0], [0,0,1]])
     sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
     cv.imshow('sheared img x', sheared_img)
     cv.waitKey(0)
     cv.destroyAllWindows()


def shearing_y_axis():
     import cv2 as cv
     import numpy as np
     img = cv.imread('Photos/park.jpg', 0)
     rows, cols = img.shape
     M = np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
     sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
     cv.imshow('sheared img y', sheared_img)
     cv.waitKey(0)
     cv.destroyAllWindows()


def contour_detection():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/park.jpg')
  cv.imshow('Original img', img)

  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  edged = cv.Canny(gray,30,300)
  contours,hierarchy = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

  cv.imshow('canny edges after contouring', edged)
  print('number of contours ='+str(len(contours)))
  cv.drawContours(img,contours,-1,(0,255,0),3)

  cv.imshow('Park',img)
  cv.waitKey(0)
  cv.destroyAllWindows()


def BGR_split():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/park.jpg')
  cv.imshow('Park',img)
  b,g,r = cv.split(img)
  cv.imshow('Blue',b)
  cv.imshow('Green',g)
  cv.imshow('Red',r)
  cv.waitKey(0)


def BGR_merge():
   import cv2 as cv
   import numpy as np
   img = cv.imread('Photos/park.jpg')
   cv.imshow('Park',img)
   blank = np.zeros(img.shape[:2], dtype='uint8')
   b,g,r = cv.split(img)
   blue = cv.merge([b,blank,blank])
   green = cv.merge([blank,g,blank])
   red = cv.merge([blank,blank,r])
   cv.imshow('Blue',blue)
   cv.imshow('Green',green)
   cv.imshow('Red',red)
   print(img.shape)
   print(b.shape)
   print(g.shape)
   print(r.shape)
   merged = cv.merge([b,g,r])
   cv.imshow('Merged img', merged)
   cv.waitKey(0)


def averaging_blur():
  import cv2 as cv
  img = cv.imread('Photos/cats.jpg')
  average = cv.blur(img, (3,3))
  cv.imshow('Average blur', average)


def median_blur():
  import cv2 as cv
  img = cv.imread('Photos/cats.jpg')
  median = cv.medianBlur(img,3)
  cv.imshow('median blur', median)


def bilateral_blur():
  import cv2 as cv
  img = cv.imread('Photos/cats.jpg')
  biblur = cv.bilateralFilter(img, 10,35,25)
  cv.imshow('BiLateral blur photo', biblur)


def bitwise():
  import cv2 as cv
  import numpy as np
  blank = np.zeros((400,400), dtype='uint8')
  rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255,-1)
  circle = cv.circle(blank.copy(), (200,200), 200, 255,-1)
  #225 is the color white and 200 is the radius of the circle
  cv.imshow('Rectangle', rectangle)
  cv.imshow('Circle', circle)

  #bitwise AND --> found intersecting regions
  #took both images, placed them on top of eachother and found the common regions
  bitwise_and = cv.bitwise_and(rectangle,circle)
  cv.imshow('Bitwise AND', bitwise_and)


  #bitwise OR ---> found non-intersecting and intersecting regions
  #took both images, placed them on top of eachother and found both common and uncommon regions
  bitwise_or = cv.bitwise_or(rectangle,circle)
  cv.imshow('Bitwise OR', bitwise_or)


  #bitwise XOR --> found non-intersection regions
  # shows the uncommon regions
  bitwise_xor = cv.bitwise_xor(rectangle,circle)
  cv.imshow('Bitwise XOR', bitwise_xor)


  #bitwise NOT --> inverts the binary color
  # inverts the binary color
  bitwise_not = cv.bitwise_not(rectangle)
  cv.imshow('Rectangle NOT', bitwise_not)

  bitwise_not = cv.bitwise_not(circle)
  cv.imshow('Circle NOT', bitwise_not)

  cv.waitKey(0)


def masking():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/cats 2.jpg')
  cv.imshow('Cats 2', img)
  blank = np.zeros(img.shape[:2], dtype='uint8')
  cv.imshow('Blank img', blank)
  mask = cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
  cv.imshow('Mask',mask)
  masked = cv.bitwise_and(img,img,mask=mask)
  cv.imshow('Masked img', masked)
  cv.waitKey(0)


def gray_histogram_computation():
  import cv2 as cv
  import numpy as np
  import matplotlib.pyplot as plt
  img = cv.imread('Photos/cats.jpg')
  cv.imshow('cats', img)
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  cv.imshow('Gray', gray)

  gray_hist = cv.calcHist([gray],[0], None, [256], [0,256])

  plt.figure()
  plt.title('Grayscale Histogram')
  plt.xlabel('Bins')
  plt.ylabel('# of pixels')
  plt.plot(gray_hist)
  plt.xlim([0,256])
  plt.show()
  cv.waitKey(0)


def BGR_histogram_computation():
   import cv2 as cv
   import numpy as np
   import matplotlib.pyplot as plt
   img = cv.imread('Photos/cats.jpg')
   cv.imshow('Cats', img)
   blank = np.zeros(img.shape[:2], dtype='uint8')
   mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
   masked = cv.bitwise_and(img,img,mask=mask)
   cv.imshow('Mask', masked)
   plt.figure()
   plt.title('Colour Histogram')
   plt.xlabel('Bins')
   plt.ylabel('# of pixels')
   colors = ('b', 'g', 'r')
   for i,col in enumerate(colors):
      hist = cv.calcHist([img], [i], mask, [256], [0,256])
      plt.plot(hist, color=col)
      plt.xlim([0,256])
   plt.show()
   cv.waitKey(0)


def simple_thresh():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/lady.jpg')
  cv.imshow('Original', img)
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  cv.imshow('Gray', gray)

  threshhold, thresh  = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
  threshhold, thresh_inv  = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
  cv.imshow('inversed thresh img', thresh_inv)
  cv.waitKey(0)


def adaptive_thresh():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/lady.jpg')
  cv.imshow('Original', img)
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  cv.imshow('Gray', gray)

  adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
  cv.imshow('Adaptive thresh img', adaptive_thresh)
  cv.waitKey(0)


def laplacian_edge_detection():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/cats.jpg')
  cv.imshow('Cats', img)

  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  cv.imshow('Gray', gray)

  lap = cv.Laplacian(gray, cv.CV_64F)
  lap = np.uint8(np.absolute(lap))
  cv.imshow('Laplacian', lap)
  cv.waitKey(0)


def sobel_edge_detection():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/cats.jpg')
  cv.imshow('Cats', img)

  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  cv.imshow('Gray', gray)

  sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
  sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
  combined_sobel = cv.bitwise_or(sobelx, sobely)

  cv.imshow('Sobel X', sobelx)
  cv.imshow('Sobel Y', sobely)
  cv.imshow('Combined Sobel', combined_sobel)
  cv.waitKey(0)

def canny_edge_detection():
  import cv2 as cv
  import numpy as np
  img = cv.imread('Photos/cats.jpg')
  cv.imshow('Cats', img)
  canny = cv.Canny('gray', 150, 175)
  cv.imshow('Canny',canny)
  cv.waitKey(0)
