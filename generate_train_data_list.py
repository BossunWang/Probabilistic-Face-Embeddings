import os
import tqdm


def write_to_file(data_dir, f):
    for dir, dirs, files in tqdm.tqdm(os.walk(data_dir)):
        for index, file in enumerate(files):
            filepath = dir + '/' + file

            if filepath.endswith(".jpg") or filepath.endswith(".JPG") or filepath.endswith(".png"):
                f.write("%s\n" % filepath)
                # print(filepath)


# generate dataset for unnormal data
print('generate dataset for training data')
f = open('data/train_list.txt', 'w')

# data_dir = '../face_dataset/ms1m_align_112/'
data_dir = '../face_dataset/CASIA-maxpy-clean_crop/'
write_to_file(data_dir, f)

data_dir = '../face_dataset/Umdfaces/'
write_to_file(data_dir, f)

f.close()

