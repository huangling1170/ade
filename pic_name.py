import os
#重新排序
# path =r"bg_pic"
# path2 =r"bg_pic2"
#
# for i,filename in enumerate(os.listdir(path2)):
#     i = i+1
#     os.replace(os.path.join(path2,filename),os.path.join(path,f"{i}.png"))

#爬虫后五花八门，删除小像素
# bg_path = "bg_pic2"
# for filname in os.listdir(bg_path):
#     size= os.path.getsize("{0}/{1}".format(bg_path, filname))
#
#     if size< 10*10:
#         os.remove("{0}/{1}".format(bg_path, filname))
#删除小像素2
# bg_path = "bg_pic2"
# bg_save = []
# for filname in os.listdir(bg_path):
#     # try:
#     background = Image.open("{0}/{1}".format(bg_path, filname))
#     shape = np.shape(background)
#     print(shape)
#     if shape[0] > 100 and shape[1] > 100 :
#         background = background
#     # else:
#     #     continue  # 跳过
#     else:
#         bg_save.append("{0}/{1}".format(bg_path, filname))
#         # os.remove("{0}/{1}".format(bg_path, filname))
# for delfil in bg_save:
#     os.remove(delfil)

#重新排序2
# class BatchRename():
#     '''
#     批量重命名文件夹中的图片文件
#
#     '''
#     def __init__(self):
#         self.path = r'E:\AIwork'  #表示需要命名处理的文件夹
#         self.save_path=r'E:\AIwork2'#保存重命名后的图片地址
#     def rename(self):
#         filelist = os.listdir(self.path) #获取文件路径
#         total_num = len(filelist) #获取文件长度（个数）
#         i = 1  #表示文件的命名是从1开始的
#         for item in filelist:
#             print(item)
#             if item.endswith('.png'):  #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
#                 src = os.path.join(os.path.abspath(self.path), item)#当前文件中图片的地址
#                 dst = os.path.join(os.path.abspath(self.save_path), ''+str(i) + '.png')#处理后文件的地址和名称,可以自己按照自己的要求改进
#                 try:
#                     os.rename(src, dst)
#                     print ('converting %s to %s ...' % (src, dst))
#                     i = i + 1
#                 except:
#                     continue
#         print ('total %d to rename & converted %d jpgs' % (total_num, i))
#
# if __name__ == '__main__':
#     demo = BatchRename()
#     demo.rename()