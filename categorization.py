import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

MOD = 0.1

def rgb_average(array):
    avg = np.mean(np.mean(array, axis = 0), axis = 0)

    return avg

def datagen():
    img_avg = []
    red = []
    green = []
    blue = []
    catlist = []
    categories = ['chaparral', 'forest', 'buildings', 'river']

    for c in categories:
        for n in range(100):
            filebase = f'UCMerced_LandUse\Images'
            filename = filebase + f"\{c}\{c}" + f"{n:02d}" + ".tif"
            img = Image.open(filename)
            imgarr = np.array(img)
            np_img = rgb_average(imgarr)
            imglist = np_img.tolist()
            red.append(imglist[0])
            green.append(imglist[1])
            blue.append(imglist[2])
            catlist.append(c)
            imglist.append(c)
            img_avg.append(imglist)
    
    df = pd.DataFrame({'red': red, 'green': green, 'blue': blue, 'category': catlist})
    
    return img_avg, df

def datavis(df):
    groups = df.groupby('category')
    ax = plt.axes(projection = '3d')

    for name, group in groups:
        ax.scatter3D(group.red, group.green, group.blue, marker = 'o', label = name)

    ax.set_xlabel('Red', fontsize = 20, rotation = 345)
    ax.set_ylabel('Green', fontsize = 20)
    ax.set_zlabel('Blue', fontsize = 20, rotation = 90)
    ax.legend()
    plt.title('3D Scatterplot of Categories in the RGB Scale')
    plt.savefig('4cat-rgbbands.jpg')
    plt.show()

def test_train(data):
    x_train, x_test = train_test_split(data, test_size = 0.2)
    
    categories = []
    grading = []
    for n in range(len(x_train)):
        categories.append(x_train[n].pop())
    for n in range(len(x_test)):
        grading.append(x_test[n].pop())
    
    clf = svm.SVC()
    clf.fit(x_train, categories)
    predicted = clf.predict(x_test)
    print(metrics.accuracy_score(grading, predicted))

def main():
    groupdata, dataframe = datagen()
    test_train(groupdata)
    datavis(dataframe)
    

if __name__ == "__main__":
    main()