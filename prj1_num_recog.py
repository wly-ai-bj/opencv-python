import cv2
import numpy as np 

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour_sort(contours):
    bbox = [cv2.boundingRect(c) for c in contours]
    (contours, bbox) = zip(*sorted(zip(contours,bbox),key = lambda b:b[1][0],reverse=False))
    return contours, bbox

#对模版进行的操作
img_tmp = cv2.imread('./dataset/template.jpg')
cv_show('img',img_tmp)
tmp_gray = cv2.cvtColor(img_tmp,cv2.COLOR_BGR2GRAY)
_, ref = cv2.threshold(tmp_gray,20,255,cv2.THRESH_BINARY_INV)
cv_show('binary',ref)
tmp_contours,_ = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#输入为二值图像，黑色为背景，白色为目标
cv2.drawContours(img_tmp,tmp_contours,-1,(0,0,255),3)
cv_show('tmplate_contour',img_tmp)

digits = {}
contours, _ = contour_sort(tmp_contours)
for (i, c) in enumerate(contours):
    (x,y,w,h) = cv2.boundingRect(c)
    roi = ref[y: y+h, x: x+w]
    roi = cv2.resize(roi,(57,88))
    digits[i] = roi 

#对输入图像进行操作
img_card = cv2.imread('./dataset/card.png')
img_card = cv2.resize(img_card,(250,200))
card_grey = cv2.cvtColor(img_card, cv2.COLOR_BGR2GRAY)

recKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
tophat = cv2.morphologyEx(card_grey, cv2.MORPH_TOPHAT,recKernel)

sobelx = cv2.Sobel(tophat,ddepth=cv2.CV_64F,dx=1,dy=0,ksize=-1)
sobelx = np.absolute(sobelx)
minval = np.min(sobelx)
maxval = np.max(sobelx)
sobelx = (255*((sobelx-minval)/(maxval-minval)))
sobelx = sobelx.astype("uint8")

sobelx = cv2.morphologyEx(sobelx,cv2.MORPH_CLOSE,recKernel)
cv_show("imagegradx",sobelx)
ret,thresh = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
card_Cnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_card,card_Cnts,-1,(0,0,255),1)
cv_show("contours",img_card) 

locs = []
for i,c in enumerate(card_Cnts):
    x,y,w,h = cv2.boundingRect(c)
    ratio = w/h 
    if ratio > 2.5 and ratio < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x,y,w,h))

locs = sorted(locs, key = lambda x:x[0])

output=[]
for (i,(gx,gy,gw,gh)) in enumerate(locs):
    groupOutput = []
    group = card_grey[gy-5:gy+gh+5,gx-5:gx+gw+5] # 在灰度图中选取roi
    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    digitCnts,hierarchy = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in digitCnts:
        (x,y,w,h) = cv2.boundingRect(c)
        roi = group[y:y+h,x:x+w]
        roi = cv2.resize(roi,(57,88))
        scores=[]
        for (digit, digitROI) in digits.items():  # 在模板预处理中建立了数值的字典类型,一个为索引、一个为值
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)  # 匹配，返回与之匹配度最高的数值
            (min_val, score, min_index, max_index) = cv2.minMaxLoc(result)  # 做10次匹配，取最大值（注意：取最大值还是最小值跟选取的模板匹配方法有关）
            scores.append(score)
        groupOutput.append(str(np.argmax(scores))) #argmax返回的是最大数的索引
    cv2.rectangle(img_card, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1) # 第一组的矩形框
    groupOutput.reverse()
    image = cv2.putText(img_card, str("".join(groupOutput)), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output.extend(groupOutput)
cv_show('image',image)
print("Credit Card #: {}".format("".join(output))) # 将output中的字符用双引号中间的符号连接起来

