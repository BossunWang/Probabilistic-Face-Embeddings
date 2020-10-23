import os


def write_to_file(data_dir, f):
    for dir, dirs, files in os.walk(data_dir):
        print('total files:', len(files))
        for index, file in enumerate(files):
            filepath = dir + '/' + file

            if filepath.endswith(".jpg") or filepath.endswith(".JPG") or filepath.endswith(".png"):
                f.write("%s\n" % filepath)
                # print(filepath)

# generate dataset for unnormal data
print('generate dataset for training data')
f = open('data/train_list.txt', 'w')

print('multi_PIE_crop_128')
data_dir = '../face_dataset/ms1m_align_112/'
write_to_file(data_dir, f)

f.close()

