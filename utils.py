import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os


class EarlyStopping:
    def __init__(self, patience=15, verbose=True, delta=0.0001, path='checkpoint.pt', 
                 monitor='val_loss', save_full_model=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = True
        self.monitor_improved = False
        self.delta = delta
        self.path = path
        self.monitor = monitor
        self.save_full_model = save_full_model

        if self.monitor in ['psnr', 'ssim', 'ms_ssim']:
            self.monitor_higher_better = True
        else:
            self.monitor_higher_better = False

    def __call__(self, current_metric, model):
        score = current_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.monitor_improved = True
        else:
            if self.monitor_higher_better:
                if score <= self.best_score + self.delta:
                    self.counter += 1
                    self.monitor_improved = False
                    if self.verbose:
                        print(f'EarlyStopping counter: {self.counter} out of {self.patience} | '
                              f'Best {self.monitor}: {self.best_score:.6f}')
                else:
                    self.best_score = score
                    self.save_checkpoint(score, model)
                    self.counter = 0
                    self.monitor_improved = True
            else:
                if score >= self.best_score - self.delta:
                    self.counter += 1
                    self.monitor_improved = False
                    if self.verbose:
                        print(f'EarlyStopping counter: {self.counter} out of {self.patience} | '
                              f'Best {self.monitor}: {self.best_score:.6f}')
                else:
                    self.best_score = score
                    self.save_checkpoint(score, model)
                    self.counter = 0
                    self.monitor_improved = True

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f'Early stopping triggered after {self.patience} epochs without improvement')

    def save_checkpoint(self, metric_value, model):
        if self.verbose:
            improved_str = 'improved' if self.best_score is None else f'improved from {self.best_score:.6f} to {metric_value:.6f}'
            print(f'{self.monitor} {improved_str}')

        if self.save_full_model:
            torch.save(model, self.path)
        else:
            torch.save(model.state_dict(), self.path)


def test_save(result_dir, generator, dataloader, device, epoch):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (lr_images, hr_images, lr_filenames, _) in enumerate(dataloader):
            lr_images = lr_images.to(device)
            fake_images = generator(lr_images)

            for i in range(lr_images.size(0)):
                generated_img = fake_images[i]
                original_filename = os.path.splitext(lr_filenames[i])[0]

                save_path = os.path.join(
                    result_dir,
                    f"epoch_{epoch}_{original_filename}.png"
                )

                save_image(generated_img, save_path)
                print(f"Saved generated image: {save_path}")

            break