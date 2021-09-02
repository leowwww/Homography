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
def mouse_action(event, x, y, flags, replace_coordinate_array):
    cv.imshow('collect coordinate', img_dest_copy)
    if event == cv.EVENT_LBUTTONUP:
        # 画圆函数，参数分别表示原图、坐标、半径、颜色、线宽(若为-1表示填充)
        # 这个是为了圈出鼠标点击的点
        cv.circle(img_dest_copy, (x, y), 2, (0, 255, 255), -1)

        # 用鼠标单击事件来选择坐标
        # 将选中的四个点存放在集合中，在收集四个点时，四个点的点击顺序需要按照 img_src_coordinate 中的点的相对位置的前后顺序保持一致
        print(f'{x}, {y}')
        replace_coordinate_array.append([x, y])


if __name__ == '__main__':
    # 首先，加载待替换的源图像，并获得该图像的长度等信息,cv.IMREAD_COLOR 表示加载原图
    img_src = cv.imread('t.png', cv.IMREAD_COLOR)
    h, w, c = img_src.shape
    # 获得图像的四个边缘点的坐标
    img_src_coordinate = np.array([[x, y] for x in (0, w - 1) for y in (0, h - 1)])
    print(img_src_coordinate)
    # cv.imshow('replace', replace)

    print("===========================")

    # 加载目标图像
    img_dest = cv.imread('1.png', cv.IMREAD_COLOR) #背景图片
    # 将源数据复制一份，避免后来对该数据的操作会对结果有影响
    img_dest_copy = np.tile(img_dest, 1)

    # 源图像中的数据
    # 定义一个数组，用来存放要源图像中要替换的坐标点，该坐标点由鼠标采集得到
    replace_coordinate = []
    cv.namedWindow('collect coordinate')
    cv.setMouseCallback('collect coordinate', mouse_action, replace_coordinate)
    while True:
        # 当采集到四个点后，可以按esc退出鼠标采集行为
        if cv.waitKey(20) == 27:
            break

    print(replace_coordinate)#替换位置
    

    replace_coordinate = np.array(replace_coordinate)
    # 根据选中的四个点坐标和代替换的图像信息完成单应矩阵
    matrix, mask = cv.findHomography(img_src_coordinate, replace_coordinate, 0)
    print(f'matrix: {matrix}')
    perspective_img = cv.warpPerspective(img_src, matrix, (img_dest.shape[1], img_dest.shape[0]))
    cv.imshow('img', perspective_img)

    # cv.imshow('threshold', threshold_img)
    # 降噪，去掉最大或最小的像素点
    retval, threshold_img = cv.threshold(perspective_img, 0, 255, cv.THRESH_BINARY)
    # 将降噪后的图像与之前的图像进行拼接
    cv.copyTo(src=threshold_img, mask=np.tile(threshold_img, 1), dst=img_dest)
    cv.copyTo(src=perspective_img, mask=np.tile(perspective_img, 1), dst=img_dest)
    cv.imshow('result', img_dest)
    cv.waitKey()
    cv.destroyAllWindows()

