import cv2
import torch
import sys

sys.path.append('/mnt/nas203/forGPU2/junghye/Deepfake_Disruption/SimSwap')
sys.path.append('/mnt/nas203/forGPU2/junghye/Deepfake_Disruption/AntiForgery')

import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from utils import lab_attack_faceswap,lab_attack_faceswap_ver2  # 적대적 공격 함수 불러오기


def lcm(a, b): 
    return abs(a * b) // fractions.gcd(a, b) if a and b else 0


# Transformers
transformer = transforms.Compose([
    transforms.ToTensor(),
])

transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()
    std = torch.tensor(std).view(1, 3, 1, 1).cuda()
    return image * std + mean


if __name__ == '__main__':
    opt = TestOptions().parse()

    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()

    with torch.no_grad():
        # Load images
        pic_a = opt.pic_a_path  # Source image
        img_a = Image.open(pic_a).convert('RGB')
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        pic_b = opt.pic_b_path  # Target image
        img_b = Image.open(pic_b).convert('RGB')
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])
        print(f'img b : {img_b.shape}, img_att : {img_att.shape}')

        # Convert to GPU tensors
        img_id = img_id.cuda()
        img_att = img_att.cuda()

        # Create latent ID
    img_id_downsample = F.interpolate(img_id, size=(112, 112))
    latend_id = model.netArc(img_id_downsample)
    latend_id = latend_id.detach().to('cpu')
    latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
    latend_id = latend_id.to('cuda')

        # Perform face swapping (no attack)
    img_fake_no_attack = model(img_id, img_att, latend_id, latend_id, True)

        # Add adversarial attack to the source image

    img_b_adv, _ = lab_attack_faceswap(img_id, img_att, model, epsilon=0.03, iter=100)

            # Perform face swapping (with attack)
    img_fake_attack = model(img_id,img_b_adv, latend_id, latend_id, True)


    img_id_denorm=denormalize(img_id)
    

    for i in range(img_id.shape[0]):
        if i == 0:
            row1 = img_id_denorm[i]
            row2 = img_att[i]
            row3_no_attack = img_fake_no_attack[i]
            row3_attack = img_fake_attack[i]
        else:
            row1 = torch.cat([row1, img_id_denorm[i]], dim=2)
            row2 = torch.cat([row2, img_att[i]], dim=2)
            row3_no_attack = torch.cat([row3_no_attack, img_fake_no_attack[i]], dim=2)
            row3_attack = torch.cat([row3_attack, img_fake_attack[i]], dim=2)

            # Concatenate results
    source=row2.detach().permute(1,2,0).to('cpu').numpy()[..., ::-1] * 255
    target=row1.detach().permute(1,2,0).to('cpu').numpy()[..., ::-1] * 255
    full_no_attack = row3_no_attack.detach().permute(1, 2, 0).to('cpu').numpy()[..., ::-1] * 255
    full_attack = row3_attack.detach().permute(1, 2, 0).to('cpu').numpy()[..., ::-1] * 255

    combined_result=np.concatenate([source,target,full_no_attack,full_attack],axis=1)

    output_path=opt.output_path+'result_combined_4_e0.03.jpg'
            # Save images
    cv2.imwrite(output_path,combined_result)

    print("Results saved at:", output_path)
    