import cv2
import numpy as np 

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def order_points(pts):
    rect = np.zeros((4,2),dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmin(diff)]
    return rect

def transform(img, pts):
    rec = order_points(pts)

    tl,tr,br,bl = rec
    w1 = np.sqrt((tr[0]-tl[0])**2+(tr[1]-tl[1])**2)
    w2 = np.sqrt((br[0]-bl[0])**2+(br[1]-bl[1])**2)
    w = max(int(w1),int(w2))

    h1 = np.sqrt((bl[0]-tl[0])**2+(bl[1]-tl[1])**2)
    h2 = np.sqrt((br[0]-tr[0])**2+(br[1]-tr[1])**2)
    h = max(int(h1),int(h2))

    dst = np.array([
        [0,0],
        [w-1,0],
        [w-1,h-1],
        [0,h-1]],dtype='float32')

    m = cv2.getPerspectiveTransform(rec,dst)
    wraped = cv2.warpPerspective(img,m,(w,h))
    return wraped

def ocr_rec():
    img_ori = cv2.imread('./dataset/xiaopiao.jpg')
    h,w,c = img_ori.shape
    wh_ratio = w/float(h)
    ratio = 500.0/h
    img_copy = img_ori.copy()
    img = cv2.resize(img_copy,(int(500*wh_ratio),500),interpolation=cv2.INTER_LINEAR)

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray,(5,5),0)   
    edge = cv2.Canny(img_gray, 75, 200)#边缘检测
    cv_show('edge',edge)

    contours,_ = cv2.findContours(edge.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[0:5]
    for c in contours:
        perimeter = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*perimeter,True)
        if len(approx) == 4:
            screenCnt = approx
            break
    cv2.drawContours(img,[screenCnt],-1,(0,0,255),2)
    cv_show('img',img)

    wraped = transform(img_ori,screenCnt.reshape(4,2)*ratio)
    wraped = cv2.cvtColor(wraped,cv2.COLOR_BGR2GRAY)
    cv_show('wraped',wraped)
    _, ref = cv2.threshold(wraped,100,255,cv2.THRESH_BINARY)
    cv_show('ref',ref)
    pass


def main():
    ocr_rec()

if __name__ == "__main__":
    main()