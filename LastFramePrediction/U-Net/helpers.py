import torch
import matplotlib.pyplot as plt
import os 


def denormalize(tensor, means=[0.4437, 0.4503, 0.2327], stds=[0.2244, 0.2488, 0.0564]):
    """
    Denormalizes a tensor of images given the means and stds.
    tensor: Tensor of images, expected shape [C, H, W] or [N, C, H, W]
    means: List of means for each channel.
    stds: List of standard deviations for each channel.
    """
    if tensor.ndim == 3:  # Single image [C, H, W]
        tensor = tensor.clone()
        for c in range(3):
            tensor[c] = tensor[c] * stds[c] + means[c]
        return tensor
    elif tensor.ndim == 4:  # Batch of images [N, C, H, W]
        tensor = tensor.clone()
        for c in range(3):
            tensor[:, c] = tensor[:, c] * stds[c] + means[c]
        return tensor
    else:
        raise ValueError("Unsupported tensor shape; expected 3 or 4 dimensions, got {}".format(tensor.ndim))
    

def plot_images(real_images, target_image, reconstructed_image, epoch, batch_idx, directory):
    """
    Plots and saves the real images, target image, and reconstructed image.
    """
    # Denormalize the images if they're in float format
    if real_images.dtype == torch.float32:
        real_images = [denormalize(img) for img in real_images]
        target_image = denormalize(target_image)
    
    # Assuming reconstructed_image is already denormalized if necessary
    # If it's also normalized, you should denormalize it similarly:
    if reconstructed_image.dtype == torch.float32:
        reconstructed_image = denormalize(reconstructed_image)

    # Create a figure with 6 subplots
    fig, axs = plt.subplots(1, 6, figsize=(18, 3))

    # Plotting the first four real images
    for i, img in enumerate(real_images):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        axs[i].imshow(img_np)
        axs[i].axis('off')  # Turn off the axis

    # Plotting the fifth real image (target)
    target_img_np = target_image.permute(1, 2, 0).cpu().numpy()
    axs[4].imshow(target_img_np)
    axs[4].axis('off')  # Turn off the axis

    # Plot the reconstructed image
    rec_img_np = reconstructed_image.permute(1, 2, 0).cpu().numpy()
    axs[5].imshow(rec_img_np)
    axs[5].axis('off')  # Turn off the axis

    # Save the plot
    plt.savefig(os.path.join(directory, f"plot_epoch{epoch+1}_batch{batch_idx+1}.png"))
    plt.close()

