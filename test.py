'''from cv2 import cv
import numpy as np'''

'''def main():

    # 1.导入图片
    img_src = cv2.imread("1.png")
    height, width = img_src.shape[:2]
    print("img width:%d height:%d" % (width, height))

    # 2.创建原图与目标图的对应点
    src_point = np.float32([[width * 0.15, 0], [width - 1, 0],
                            [0, height - 1], [width * 0.85, height - 1]])

    dst_point = np.float32([[0, 0], [width - 1, 0],
                            [0, height - 1], [width - 1, height - 1]])

    # 3.获取透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_point, dst_point)

    # 4.执行透视变换
    img_dst = cv2.warpPerspective(img_src, perspective_matrix, (width, height))

    # 5.显示结果
    cv2.imshow("img_src", img_src)
    cv2.imshow("img_dst", img_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()'''
import cv2 as cv
import numpy as np


# 鼠标操作，鼠标选中源图像中需要替换的位置信息
def mouse_action(event, x, y, flags, replace_coordinate_array ):
    cv.imshow('collect coordinate', img_dest_copy)
    if event == cv.EVENT_LBUTTONUP:
        cv.circle(img_dest_copy, (x, y), 2, (0, 255, 255), -1)
        print(f'{x}, {y}')
        replace_coordinate_array.append([x, y])
def takeSecond(elem):
    return elem[1]
def main(plane_path , model_img_path):
    img_src = cv.imread(plane_path, cv.IMREAD_COLOR)
    h, w, c = img_src.shape
    plane = np.array([[x, y] for x in (0, w - 1) for y in (0, h - 1)])
    def mouse_action(event, x, y, flags, replace_coordinate_array ):
        cv.imshow('collect coordinate', img_dest_copy)
        if event == cv.EVENT_LBUTTONUP:
            cv.circle(img_dest_copy, (x, y), 2, (0, 255, 255), -1)
            print(f'{x}, {y}')
            replace_coordinate_array.append([x, y])
    img_dest = cv.imread(model_img_path, cv.IMREAD_COLOR) #背景图片
    img_dest_copy = np.tile(img_dest, 1)
    replace_coordinate = []
    cv.namedWindow('collect coordinate')
    cv.setMouseCallback('collect coordinate', mouse_action, replace_coordinate)
    while True:
        # 当采集到四个点后，可以按esc退出鼠标采集行为
        if cv.waitKey(20) == 27:
            break
    replace_coordinate.sort(key = takeSecond)
    replace_coordinate = np.array(replace_coordinate)
    matrix, mask = cv.findHomography(plane, replace_coordinate, 0)
    print(f'matrix: {matrix}')
    perspective_img = cv.warpPerspective(img_src, matrix,(img_dest.shape[1], img_dest.shape[0]))#(img_dest.shape[1], img_dest.shape[0])
    cv.imshow('img', perspective_img)

    retval, threshold_img = cv.threshold(perspective_img, 0, 255, cv.THRESH_BINARY)
    # 将降噪后的图像与之前的图像进行拼接
    cv.copyTo(src=threshold_img, mask=np.tile(threshold_img, 1), dst=img_dest)
    cv.copyTo(src=perspective_img, mask=np.tile(perspective_img, 1), dst=img_dest)
    cv.imshow('result', img_dest)
    cv.waitKey()
    cv.destroyAllWindows()
            

if __name__ == '__main__':
    main('w.png' , '1.png')
