import os
import xml.etree.ElementTree as ET


def file_name(path):
    files_name = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                t = os.path.splitext(file)[0]
                files_name.append(t)  # 将所有的文件名添加到L列表中
    return files_name  # 返回L列表

# 获取所有分类
def get_class(path, files_name):
    class_list = []
    for n in files_name:
        f_dir = path + "\\" + n + ".xml"
        in_file = open(f_dir, encoding='UTF-8')
        filetree = ET.parse(in_file)
        in_file.close()
        root = filetree.getroot()
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in class_list or int(difficult) == 1:
                class_list.append(cls)
    return class_list


def xml2dota(path, save_path, class_list):
    filetree = ET.parse(path)
    root = filetree.getroot()
    lins = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        obj_id = class_list.index(obj_name)
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('x1').text)
        y1 = int(bbox.find('y1').text)
        x2 = int(bbox.find('x2').text)
        y2 = int(bbox.find('y2').text)
        x3 = int(bbox.find('x3').text)
        y3 = int(bbox.find('y3').text)
        x4 = int(bbox.find('x4').text)
        y4 = int(bbox.find('y4').text)
        lins.append([obj_id, x1, y1, x2, y2, x3, y3, x4, y4])
    with open(save_path, "w") as fil:
        for lin in lins:
            for x in lin:
                fil.write(str(x)+" ")
            fil.write("\n")


if __name__ == "__main__":
    input_dir = r"H:\YOLO-Project\data_tools\DataXml"
    out_dir = r"H:\YOLO-Project\data_tools\DataTxt"
    filelist = file_name(input_dir)
    # 获取类别并保存
    cls_list = get_class(input_dir, filelist)
    with open(out_dir+"/"+'classes.txt', 'w') as fil:
        fil.write("nc: "+str(len(cls_list))+'\n')
        for n in cls_list:
            fil.write(n+'\n')
    for n in filelist:
        xml2dota(input_dir+"/"+n+'.xml', out_dir+"/"+n+".txt", cls_list)





