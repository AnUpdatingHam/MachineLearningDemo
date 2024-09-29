import os
import time
import json
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

file_dir = 'G:\\dataset\\mchar\\'  # 数据集路径

train_dict_path = 'mchar_train.json'  # 训练json数据路径
train_img_dir_path = 'mchar_train'  # 训练图片存储路径

test_dict_path = 'mchar_test.json'  # 测试json数据路径
test_img_dir_path = 'mchar_test'  # 测试图片存储路径

IMAGE_SIZE = (16, 16)  # 训练图片大小
H, W = IMAGE_SIZE[0] - 1, IMAGE_SIZE[1] - 1
TRAIN_IMAGE_CNT = 1  # 训练集图片数
TEST_IMAGE_CNT = TRAIN_IMAGE_CNT * 0.25  # 测试集图片数


# 自适应二值化
def binarize(img_array):
    _, binary_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 通过四个角的权重计算是否需要翻转黑白像素（尽量保证背景是白色，字是黑色）
    weight = (int(binary_array[0, 0]) + int(binary_array[H, 0])
              + int(binary_array[0, W]) + int(binary_array[H, W]))
    if weight < 510:
        cond = (binary_array == 0)
        binary_array[...] = 0
        binary_array[cond] = 255

    binary_array = binary_array.astype(np.uint8)
    # debug
    # binary_img = Image.fromarray(binary_array)  #  debug输出二值化后的图片
    # binary_img.show()
    return binary_array


# 图片转灰度图，裁剪图片内的数字，拉伸为16x16矩阵，返回裁剪后的像素列表
def crop_number(img, info):
    img_gray = img.convert('L')  # 转换为灰度图像
    sz = len(info['label'])
    X_clip = np.array([])
    y_clip = np.array(info['label'])
    for i in range(sz):
        img_clip = img_gray.crop((info['left'][i], info['top'][i],
                                  info['left'][i] + info['width'][i],
                                  info['top'][i] + info['height'][i]))
        img_clip = img_clip.resize(IMAGE_SIZE, Image.NEAREST)

        pixel_array = binarize(np.array(img_clip))
        pixel_array = pixel_array.flatten()

        X_clip = np.append(X_clip, pixel_array)
    return X_clip, y_clip


def get_json_dict(filepath):
    with open(file_dir + filepath, 'r', encoding='utf-8') as json_file:
        # 将JSON文件的内容加载到字典中
        data = json.load(json_file)
        return data


def get_data(dir_path, info_dict, image_cnt):
    X = np.array([])
    y = np.array([])
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    # 遍历dir_path下的所有文件
    for root, dirs, files in os.walk(file_dir + dir_path):
        for file in files:
            # 检查文件扩展名是否是图片格式
            if any(file.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(root, file)
                file_name = os.path.basename(full_path)
                img = Image.open(full_path)
                X_clip, y_clip = crop_number(img, info_dict[file_name])
                X = np.append(X, X_clip)
                y = np.append(y, y_clip)

                image_cnt -= 1
                if image_cnt <= 0:
                    break
    X = X.reshape(-1, IMAGE_SIZE[0] * IMAGE_SIZE[1])
    return X, y


def decline_features(x_train, x_test):
    pca = PCA(n_components=0.95)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca


train_info_dict = get_json_dict(train_dict_path)
test_info_dict = get_json_dict(test_dict_path)

X_train, y_train = get_data(train_img_dir_path, train_info_dict, TRAIN_IMAGE_CNT)
X_test, y_test = get_data(test_img_dir_path, test_info_dict, TEST_IMAGE_CNT)

X_train_pca, X_test_pca = decline_features(X_train, X_test)

start_time = time.time()


# 单次训练
def once_train(param_C):
    svc = SVC(kernel='rbf', C=param_C)
    return svc


# 获取最佳C值
def get_best_C():
    svc = SVC()
    param_grid = {'C': np.linspace(5, 20, 100), 'kernel': ['rbf']}
    # param_grid = {'C': np.arange(5, 20), 'kernel': ['rbf']}
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_pca, y_train)

    print("最佳C值:", grid_search.best_params_['C'])
    print("最佳分数:", grid_search.best_score_)

    best_svc = SVC(C=grid_search.best_params_['C'], kernel='rbf')
    return best_svc


model = once_train(5.3)
# model = get_best_C()
model.fit(X_train_pca, y_train)

end_time = time.time()

print("测试集分数:", model.score(X_test_pca, y_test))
print(f"Model run time: {(end_time - start_time):.6f}s")
