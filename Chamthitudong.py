import cv2                           #Nguyễn Ngọc Long 0969516270
import numpy as np
import pandas as pd
from datetime import datetime
import time
import  os

print('Nhap duong dan den folder da tao:', end=' ')
link_folder = input()
list_link = os.listdir(link_folder)

for z in list_link:
    if z == 'Baithi':
        link = link_folder+'\Baithi'
    if z == 'diemthi.csv':
        link_grade = link_folder+'\diemthi.csv'
    if z == 'dapan.csv':
        link_key = link_folder+'\dapan.csv'
    if z == 'Baicham':
        link_save = link_folder+'\Baicham'
    if z =='Bailoi':
        link_save_fail = link_folder+'\Bailoi'
link_element = os.listdir(link)


widthimg = 600
heighimg = 600
questions =120
choices = 4
check_loi = 0
#chuyển đổi đáp án theo hàng dọc về hàng ngang
def transform(key):
    key_transform = []
    for i in range(0,30):
        for j in range(i,120,30):
            key_transform.append(key[j])
    return key_transform

#tính tọa độ các đáp án theo 4 cột
def processing(list):
    check = 0
    list_processing = []
    for i in range(len(list)):
        if check == 3:
            list_processing.append(list[i]+16)
        elif check == 2:
            list_processing.append(list[i] + 11)
        elif check == 1:
            list_processing.append(list[i] + 6)
        elif check == 0:
            list_processing.append(list[i] + 1)
        if check == 3:
            check=0
        else:
            check+=1
    return list_processing

#lấy ra contour và sắp xếp giảm dần
def rectContours(Contours):
    rectCon = []
    for i in Contours:
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key = cv2.contourArea, reverse = True)
    return rectCon


def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

#Tìm ra 4 góc phần cần lấy
def reorder(myPoints):
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

#tách ảnh ra các ảnh con
def splitBoxes(img,questions,choices):
    rows = np.vsplit(img,questions)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,choices)
        for box in cols:
            boxes.append(box)
    return boxes

#tô màu đáp án
def showAnswers(img, grading, key, questions, choices):
    secW = int(img.shape[1] /choices )
    secH = int(img.shape[0] /questions )
    i=0
    check = 0
    key_processed = processing(key)
    for x in range(len(key_processed)):
            cX = (key_processed[x] * secW) + secW // 2
            cY = (i * secH) + secH // 2
            if grading[x] == 1:
                corlor = (0,255,0)
            elif grading[x] == 0:
                corlor = (0, 0, 254)
                cv2.circle(img, (cX, cY), 8, corlor, 2)
            elif grading[x] == 3:
                corlor = (0, 153, 255)
            cv2.circle(img, (cX, cY), 8, corlor, 2)
            if check == 3:
                check =0
                i+=1
            else:
                check+=1
    return img

#đọc và lấy ra đáp án
def read_file(Id,link_key):
    df =  pd.read_csv(link_key)
    key = df[Id].tolist()
    return key

#chuyển đáp án abcd lấy từ excel thành 0,1,2,3
def change(arr):
    key = []
    for i in arr:
        if i =='A' or i == 'a':
            key.append(0)
        elif i =='B' or i == 'b':
            key.append(1)
        elif i == 'C' or i == 'c':
            key.append(2)
        elif i == 'D' or i =='d':
            key.append(3)
    return key

#chuyển đáp án 0,1,2,3 về abcd
def change_(arr):
    answers=[]
    answers_str = ''
    for i in arr:
        if i == 0:
            answers.append('A')
        if i == 1:
            answers.append('B')
        if i ==2:
            answers.append('C')
        if i ==3:
            answers.append('D')
        if i ==8:
            answers.append('-')
        if i == 9:
            answers.append('*')
    for i in answers:
        answers_str += i
    return answers_str

#kiểm tra mã đề
def check(Id_test,link_key):
    df = pd.read_csv(link_key)
    header = df.head(0)
    return Id_Test in header

#lưu kết quả vào file
def write_file(Id_Person,Id_test,status,score_physics,score_chemistry,score_Biology,time,ans,count_0,count_1, count_3,path,path1,link_grade):
    df = pd.read_csv(link_grade)
    Id_list = df['Sbd'].tolist()
    Id_list.append(Id_Person)
    Id_test_list = df['MaDe'].tolist()
    Id_test_list.append(Id_test)
    status_list = df['TinhTrang'].tolist()
    status_list.append(status)
    score_physics_list = df['DiemLy'].tolist()
    score_physics_list.append(score_physics)
    score_chemistry_list = df['DiemHoa'].tolist()
    score_chemistry_list.append(score_chemistry)
    score_Biology_list = df['DiemSinh'].tolist()
    score_Biology_list.append(score_Biology)
    time_list = df['ThoiGianCham'].tolist()
    time_list.append(time)
    count_0_list = df['CauSai'].tolist()
    count_0_list.append(count_0)
    count_1_list = df['CauDung'].tolist()
    count_1_list.append(count_1)
    count_3_list = df['ChuaTo/2DapAn/ToNhat'].tolist()
    count_3_list.append(count_3)
    ans_list = df['DapAnHsChon'].tolist()
    ans_list.append(ans)
    path_list = df['DuongDan'].tolist()
    path_list.append(path)
    path1_list = df['DuongDanCham'].tolist()
    path1_list.append(path1)
    df = pd.DataFrame({'Sbd' :Id_list,
                       'MaDe' : Id_test_list,
                       'TinhTrang' : status_list,
                       'DiemLy' :score_physics_list,
                       'DiemHoa':score_chemistry_list,
                       'DiemSinh': score_Biology_list,
                       'CauDung' :count_1_list,
                       'CauSai'  : count_0_list,
                       'ChuaTo/2DapAn/ToNhat': count_3_list,
                       'ThoiGianCham' : time_list,
                       'DapAnHsChon': ans_list,
                       'DuongDan': path_list,
                       'DuongDanCham': path1_list})
    df.to_csv(link_grade)

time_pre = time.time()

for count_number in link_element:
    path = link+'\{}'.format(count_number)
    #tiền xử lý
    img = cv2.imread(path)
    #img = cv2.resize(img,(widthimg,heighimg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgThres = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,3)

    #tìm contours cần lấy
    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectCon = rectContours(contours)
    answers = getCornerPoints(rectCon[0])
    answers = reorder(answers)
    sbd = getCornerPoints(rectCon[2])
    sbd = reorder(sbd)
    made = getCornerPoints(rectCon[3])
    made = reorder(made)

    #bóc lấy mã đề
    ptst1 = np.float32(made)
    ptst2 = np.float32([[0,0],[widthimg,0],[0,heighimg],[widthimg,heighimg]])
    matrix_id_test = cv2.getPerspectiveTransform(ptst1,ptst2)
    img_test_wrap = cv2.warpPerspective(img, matrix_id_test,(widthimg,heighimg))

    #bóc lấy phần đáp án
    ptsa1 = np.float32(answers)
    ptsa2 = np.float32([[0,0],[widthimg,0],[0,heighimg],[widthimg,heighimg]])
    matrix_answer = cv2.getPerspectiveTransform(ptsa1,ptsa2)
    img_answers_wrap = cv2.warpPerspective(img, matrix_answer,(widthimg,heighimg))

    #bóc lấy  sbd
    ptsp1 = np.float32(sbd)
    ptsp2 = np.float32([[0,0],[widthimg,0],[0,heighimg],[widthimg,heighimg]])
    matrix_id_person = cv2.getPerspectiveTransform(ptsp1,ptsp2)
    img_person_wrap = cv2.warpPerspective(img, matrix_id_person,(widthimg,heighimg))

    #tách các ô trong mã đề ra ảnh con
    imgIdTesstWarpGray = cv2.cvtColor(img_test_wrap, cv2.COLOR_BGR2GRAY)
    imgIdTesstThresh = cv2.threshold(imgIdTesstWarpGray,120,255,cv2.THRESH_BINARY_INV)[1]
    boxes_Id_Test = splitBoxes(imgIdTesstThresh,10,3)

    #tách ra a,b,c,d là các ảnh con
    imgAnswerWarpGray = cv2.cvtColor(img_answers_wrap, cv2.COLOR_BGR2GRAY)
    imgAnswerThresh = cv2.threshold(imgAnswerWarpGray,150,255,cv2.THRESH_BINARY_INV)[1]
    boxes_Answer = splitBoxes(imgAnswerThresh,30,20)
    answers_list = []
    for i in range(0,len(boxes_Answer),1):
        if i % 5 != 0 or i % 5 != 0:
            answers_list.append(boxes_Answer[i])


    #tách các ô trong  sbd ra ảnh con
    imgPersonTesstWarpGray = cv2.cvtColor(img_person_wrap, cv2.COLOR_BGR2GRAY)
    imgPersonTesstThresh = cv2.threshold(imgPersonTesstWarpGray,120,255,cv2.THRESH_BINARY_INV)[1]
    boxes_Id_Person = splitBoxes(imgPersonTesstThresh,10,6)

    #đếm các pixel khác 0 của ô mã đề
    count_pixels_id_test=np.zeros((10,3))
    countC_test = 0
    countR_test = 0
    for image in boxes_Id_Test:
        total_pixels_test = cv2.countNonZero(image[5:55,30:180])  # đếm các pixels khác 0
        count_pixels_id_test[countR_test, countC_test] = total_pixels_test
        countC_test += 1
        if countC_test == 3:
            countR_test += 1
            countC_test = 0

    # đếm các pixcel khác 0 các 4 đáp án
    count_pixels=np.zeros((questions,choices))
    countC = 0
    countR = 0
    for image in answers_list:
        total_pixels = cv2.countNonZero(image[2:18,2:28])  # đếm các pixels khác 0
        count_pixels[countR, countC] = total_pixels
        countC += 1
        if countC == 4: # 4 đáp án để chọn
            countR += 1
            countC = 0

    #đếm các pixel khác 0 của ô sbd
    count_pixels_id_person=np.zeros((10,6))
    countC_person = 0
    countR_person = 0
    for image in boxes_Id_Person:
        total_pixels_person = cv2.countNonZero(image[5:55,10:90])  # đếm các pixels khác 0
        count_pixels_id_person[countR_person, countC_person] = total_pixels_person
        countC_person += 1
        if countC_person == 6:
            countR_person += 1
            countC_person = 0
    #tìm ra mã đề đã tô
    temp = []
    Id_Test=''
    for i in range(3):
        arr_test = count_pixels_id_test[:,i]
        max_values_test = np.where(arr_test == np.amax(arr_test))
        temp.append(max_values_test[0][0])
    for i in temp:
        Id_Test += str(i)

    # tìm ra đáp án đã tô
    your_choices_before=[]
    for x in range (0,questions):
        check_max = 0
        check_min = 0
        arr = count_pixels[x]
        for i in arr:
            if i >130:
                check_max +=1
            if i < 75:
                check_min+=1
        if check_max <2 :
            if check_min != 4:
                max_values = np.where(arr == np.amax(arr))
                your_choices_before.append(max_values[0][0])
            else:
                your_choices_before.append(8)
        else:
            your_choices_before.append(9)  #8 = 0 tô, 9 = tô 2 đáp án
    #print(count_pixels)
    count = 0
    your_choices_after = [] #chuyển đáp án theo hàng ngang về hàng dọc
    for i in range(0,4):
        for j in range(i,120,4):
            your_choices_after.append(your_choices_before[j])
            count+=1

    #tìm sbd đã tô
    temp = []
    Id_Person=''
    for i in range(6):
        arr_person = count_pixels_id_person[:,i]
        max_values_person = np.where(arr_person == np.amax(arr_person))
        temp.append(max_values_person[0][0])
    for i in temp:
        Id_Person += str(i)

    # khởi tạo điểm mặc định nếu không chấm được còn truyền được vào hàm
    score_physics = -1
    score_chemistry = -1
    score_Biology = -1
    #kiểm tra mã đề
    check_grading = 0
    now = datetime.now()
    dt_string = now.strftime("""%d/%m/%Y %H:%M:%S""")
    if check(Id_Test,link_key) == True:

        #Lấy và biến đổi đáp án từ file
        key = read_file(Id_Test,link_key)
        key = change(key)

        #Chấm điểm 1 đúng 2 sai 3 phạp quy
        grading=[]
        for x in range(0,questions):
            if key[x] == your_choices_after[x]:
                grading.append(1)
            elif your_choices_after[x] == 9:
                grading.append(3)
            elif your_choices_after[x] == 8:
                grading.append(3)
            else:grading.append(0)
        score_physics = round((grading[0:40].count(1)/40)*10,2)
        score_chemistry = round((grading[40:80].count(1) / 40) * 10, 2)
        score_Biology = round((grading[80:120].count(1) / 40) * 10, 2)
        score = str(score_physics)+'/'+str(score_chemistry)+'/'+str(score_Biology)
        count_0 = grading.count(0)
        count_1 = grading.count(1)
        count_3 = grading.count(3)

        # tô màu lên đáp án
        key = transform(key)
        grading = transform(grading)
        your_choices_before = processing(your_choices_before)
        img_black = np.zeros_like(img_answers_wrap)
        img_black = showAnswers(img_black, grading, key, 30, 20)
        imvmatrix = cv2.getPerspectiveTransform(ptsa2, ptsa1)
        imgInvWarp = cv2.warpPerspective(img_black, imvmatrix, (1646, 2331))
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
        ans = change_(your_choices_after)
        #Viết thời gian vào giấy
        cv2.putText(imgFinal, str(dt_string), (1200, 1870), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        #viết điểm lên giấy
        cv2.putText(imgFinal, str(score), (1230,1470), cv2.FONT_HERSHEY_COMPLEX,1, (0, 0, 255),1 )

        #viết số câu đúng lên ảnh
        cv2.putText(imgFinal, str(count_1) + '/' + str(len(grading)), (1270, 1300), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)

        check_grading += 1
    else:
        count_0 = 0
        count_1 = 0
        count_3 = 0
        ans = change_(your_choices_after)

    if check_grading ==1:
        status = 'Thanh Cong'
        print('{}) Success'.format(count_number))
    else:
        status = 'Loi'
        check_loi +=1
        print('{}) Fail'.format(count_number))
        cv2.imwrite('{}\{}'.format(link_save_fail, count_number), imgFinal)

    #viết mã đề vào giấy
    cv2.putText(imgFinal, Id_Test, (1291,1130), cv2.FONT_HERSHEY_COMPLEX, 1.5,(0, 0, 255), 2)

    #viết ra thời gian chạy code
    if count_number == link_element[-1]:
        time_run = (time.time() - time_pre)
        print('  Finished in {}s '.format(round(time_run,2)))
        print('Fail:{}/{}: '.format(check_loi, len(link_element)))


    #viết thông tin ra excel
    path1='{}\{}'.format(link_save,count_number)
    write_file(Id_Person,Id_Test,status,score_physics,score_chemistry,score_Biology,dt_string,ans,count_0,count_1,count_3,path,path1,link_grade)
    cv2.imwrite('{}\{}'.format(link_save,count_number),imgFinal)
    '''cv2.imshow('', img_person_wrap)
    cv2.waitKey(0)
    print(boxes_Id_Person[1].shape)
    for im in range(len(boxes_Id_Person)):
        print(im)
        cv2.imshow('',boxes_Id_Person[im][5:55,10:90])
        cv2.waitKey(0)'''
    # Nguyễn Ngọc Long 0969516270