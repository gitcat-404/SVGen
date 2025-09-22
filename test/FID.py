from pytorch_fid import fid_score

def caculate_fid(real_images_path,generated_images_path):
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_images_path],
        batch_size=50,  
        device='cuda',  
        dims=2048
    )
    return fid_value

