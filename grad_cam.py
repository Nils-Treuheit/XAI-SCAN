import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import matplotlib.pyplot as plt

def grad_cam(model,input_rgb_img, input_img_arr):
    target_layers = [model.backbone.layer4[-1]]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    input_rgb_img = np.float32(input_rgb_img) / 255
    input_tensor = preprocess_image(input_rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    input_tensor= torch.tensor(np.array([input_img_arr]),device = torch.device("cuda"))
    
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    cam = GradCAM(model=model.backbone,
             target_layers=target_layers,
             use_cuda=True)
    
    grayscale_cams = cam(input_tensor=input_tensor)

    print(np.max(input_rgb_img))
    cam_image = show_cam_on_image(input_rgb_img, grayscale_cams[0, :], use_rgb=True)
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2)

    # Plot the first image
    axes[0].imshow(input_rgb_img)
    axes[0].set_title('Original RGB Image')

    # Plot the second image
    axes[1].imshow(cam_image)
    axes[1].set_title('GradCAM Focus Heatmap')

    # Adjust the spacing between subplots
    plt.tight_layout()
    
    # Save the image to a JPEG file
    plt.savefig('singleinstance_gradcam.jpg', format='jpeg')
    plt.show()