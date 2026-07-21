

import os
import shutil

root = "/root/autodl-fs/AAAI-SemDINO/LandsatSCD"

for mode in ["train", "val", "test"]:
    txt = os.path.join(root, f"{mode}_list.txt")
    with open(txt) as f:
        names = [line.strip() for line in f if line.strip()]


    for folder in ["A", "B", "labelA", "labelB"]:
        os.makedirs(os.path.join(root, mode, folder), exist_ok=True)

    for name in names:
        shutil.copy2(os.path.join(root, "A", name), os.path.join(root, mode, "A", name))
        shutil.copy2(os.path.join(root, "B", name), os.path.join(root, mode, "B", name))
        

        shutil.copy2(os.path.join(root, "labelA", name), os.path.join(root, mode, "labelA", name))
        shutil.copy2(os.path.join(root, "labelB", name), os.path.join(root, mode, "labelB", name))






'''
RGB
'''

# import os
# from shutil import copyfile

# def main():
#     src_dir = '~/SECOND'
#     train_dir = '~/train_info.txt'
#     val_dir = '~/val_info.txt'

#     train_info = open('train_info.txt', 'r')
#     train_list = train_info.readlines()
#     val_info = open('val_info.txt', 'r')
#     val_list = val_info.readlines()
#     dir_names = ['im1', 'im2', 'label1', 'label2', 'label1_rgb', 'label2_rgb']

#     count = 0
#     for it in train_list:
#         _, it_name = os.path.split(it.strip())
#         for dir_name in dir_names:
#             dst_dir = os.path.join(train_dir, dir_name)
#             if not os.path.exists(dst_dir): os.makedirs(dst_dir)
#             src_path = os.path.join(src_dir, dir_name, it_name)
#             dst_path = os.path.join(dst_dir, it_name)
#             copyfile(src_path, dst_path)
#         count += 1
#         if not count % 100: print('%d/%d images saved.' % (count, len(train_list)))

#     count = 0
#     for it in val_list:
#         _, it_name = os.path.split(it.strip())
#         for dir_name in dir_names:
#             dst_dir = os.path.join(val_dir, dir_name)
#             if not os.path.exists(dst_dir): os.makedirs(dst_dir)
#             src_path = os.path.join(src_dir, dir_name, it_name)
#             dst_path = os.path.join(dst_dir, it_name)
#             copyfile(src_path, dst_path)
#         count += 1
#         if not count % 100: print('%d/%d images saved.' % (count, len(train_list)))


# if __name__ == '__main__':
#     main()
