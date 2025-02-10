import os
from PIL import Image

# 输入图像和掩码文件夹路径
input_img_folder = r'D:\DeepLearning\airs\data\LoveDA\Val\Urban\images_png'
input_mask_folder = r'D:\DeepLearning\airs\data\LoveDA\Val\Urban\masks_png_convert'

# 输出文件夹路径
output_img_folder = r'D:\DeepLearning\airs\data\LoveDA\Val\Urban\images_512'
output_mask_folder = r'D:\DeepLearning\airs\data\LoveDA\Val\Urban\masks_512'

# 确保输出文件夹存在
os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# 获取图像文件夹中的所有图片文件
for filename in os.listdir(input_img_folder):
    if filename.endswith('.png'):
        # 获取图像和掩码的路径
        img_path = os.path.join(input_img_folder, filename)
        mask_path = os.path.join(input_mask_folder, filename)

        # 确保对应的掩码存在
        if os.path.exists(mask_path):
            # 打开图像和掩码
            img = Image.open(img_path)
            mask = Image.open(mask_path)

            # 确保图像是1024x1024
            if img.size == (1024, 1024):
                for i in range(2):
                    for j in range(2):
                        left = j * 512
                        upper = i * 512
                        right = left + 512
                        lower = upper + 512

                        # 裁剪图像和掩码
                        cropped_img = img.crop((left, upper, right, lower))
                        cropped_mask = mask.crop((left, upper, right, lower))

                        # 保存裁剪后的图像和掩码到不同的文件夹
                        cropped_img_filename = f"{os.path.splitext(filename)[0]}_{i * 2 + j}.png"
                        cropped_mask_filename = f"{os.path.splitext(filename)[0]}_{i * 2 + j}.png"

                        cropped_img.save(os.path.join(output_img_folder, cropped_img_filename))
                        cropped_mask.save(os.path.join(output_mask_folder, cropped_mask_filename))

print("图像和掩码裁剪完毕，已分别保存到不同的文件夹！")