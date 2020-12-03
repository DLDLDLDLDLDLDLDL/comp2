from efficientdet import Efficientdet
from PIL import Image
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

efficientdet = Efficientdet()

test_img_files = open('./pascal_voc_testing_data.txt')
test_img_dir = './VOCdevkit/VOC2007/JPEGImages/'
test_images = []

for line in test_img_files:
    line = line.strip()
    ss = line.split(' ')
    test_images.append(ss[0])

output_file = open('./test_prediction.txt', 'w')

res_list = []

for img_name in test_images:
    img = test_img_dir + img_name
    #print(img)
    try:
        image = Image.open(img)
        #print(image)
    except:
        print('Can not find image!')
        continue
    else:
        x0, x1, y0, y1, label, conf = efficientdet.output_txt(image)
        #print("img: ", img, x0, x1, y0, y1, label, conf)
        res = ""
        for i, c in enumerate(label):
            res += " %d %d %d %d %d %f" %(x0[i], x1[i], y0[i], y1[i], label[i], conf[i])
            
        #print("res: ", res)
        res_list.append(img_name+res)


for i, c in enumerate(res_list):
    output_file.write(res_list[i] + "\n")

output_file.close()