import torch
from torchvision import transforms
from backbone.model_irse import IR_101, l2_norm
from PIL import Image
import numpy as np
import os
import tqdm


def read_img(img, transform):
    pil_image = Image.open(os.path.join(img))
    image_tensor = transform(pil_image)
    return image_tensor


def feature_extract(data_list, org_folder, extract_folder):
    f = open(data_list, 'r')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = [112, 112]

    backbone_path = './backbone/CurricularFace_Backbone.pth'
    backbone_model = IR_101(INPUT_SIZE).to(device)

    checkpoint = torch.load(backbone_path, map_location=lambda storage, loc: storage)
    if 'state_dict' not in checkpoint:
        backbone_model.load_state_dict(checkpoint)

    backbone_model.eval()

    data_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE[0], INPUT_SIZE[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    for file_path in tqdm.tqdm(f):
        file_path = file_path.replace('\n', '')
        img_tensor = read_img(file_path, data_transform)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            mu, conv_final = backbone_model(img_tensor)
            mu = l2_norm(mu)

        mu = mu.cpu().data.numpy()
        conv_final = conv_final.cpu().data.numpy()

        sub_dir = file_path.split('/')[-2]

        new_dir = extract_folder + sub_dir
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        new_file_path = file_path.replace(org_folder, extract_folder)
        mu_file_path = new_file_path.replace('.jpg', '_mu.npy')
        conv_final_file_path = new_file_path.replace('.jpg', '_conv_final.npy')
        np.save(mu_file_path, mu)
        np.save(conv_final_file_path, conv_final)

    f.close()


if __name__ == "__main__":
    org_folder = '../face_dataset/ms1m_align_112/'
    extract_folder = '../face_dataset/ms1m_align_112_feature/'
    feature_extract('data/train_list.txt', org_folder, extract_folder)