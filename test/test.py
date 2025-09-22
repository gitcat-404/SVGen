from hps import calculate_hps
from aesthetic import compute_aesthetic_scores
from FID import caculate_fid
from CLIP_score_I2I import calculate_I2I_clip_score
from CLIP_score_T2I import calculate_T2I_clip_score

# 定义路径和文件
image_ground_truth = "test/PNG-test"
image_generated_folder = "your_generated_image_folder"
mlp_weights_path = "test/pretrain_weight/sac+logos+ava1-l14-linearMSE.pth"
hpc_path = "test/pretrain_weight/hpc.pt"
desc_csv = "test/color_test.csv"

# 计算分数
fid_score = caculate_fid(image_ground_truth, image_generated_folder)
clip_I2I = calculate_I2I_clip_score(image_ground_truth, image_generated_folder)
clip_T2I = calculate_T2I_clip_score(image_generated_folder, desc_csv)
aesthetic_score = compute_aesthetic_scores(image_generated_folder, mlp_weights_path)
average_hps = calculate_hps(hpc_path, image_generated_folder, desc_csv)
print("---------------------------------")
print(f"folder:{image_generated_folder}")
print(f"FID Score: {fid_score}")
print(f"CLIP I2I Score: {clip_I2I}")
print(f"CLIP T2I Score: {clip_T2I}")
print(f"Aesthetic Score: {aesthetic_score}")
print(f"Average HPS: {average_hps}")
