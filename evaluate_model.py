import os
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.utils.data as Data

from ConSinGAN.config import get_arguments
import ConSinGAN.functions as functions
import ConSinGAN.models as models
from ConSinGAN.imresize import imresize, imresize_to_shape

class Video_dataset(Data.Dataset):
    """Faces."""

    def __init__(self, root_dir, size, ext, opt):
        self.root_dir = root_dir
        self.size = size
        self.ext = ext

        self.data = {}

        for j in range(self.size):
            img_name = os.path.join(self.root_dir, str(j) + self.ext)
            image = functions.read_image_dir(img_name, opt)
            image = functions.adjust_scales2image(image, opt)
            imgs = functions.create_reals_pyramid(image, opt)

            for i in range(len(imgs)): 
                imgs[i] = imgs[i].view(3,imgs[i].shape[2],imgs[i].shape[3])

            self.data[j] = imgs

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return tuple(self.data[idx])

    def __getimageorg__(self,opt):
        img_name = os.path.join(self.root_dir, '0' + self.ext)
        image = functions.read_image_dir(img_name, opt)

        return image

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def generate_samples(netG, reals_shapes, noise_amp, scale_w=1.0, scale_h=1.0, reconstruct=False, n=50):
    if reconstruct:
        reconstruction = netG(fixed_noise, reals_shapes, noise_amp)
        if opt.train_mode == "generation" or opt.train_mode == "retarget":
            functions.save_image('{}/reconstruction.jpg'.format(dir2save), reconstruction.detach())
            functions.save_image('{}/real_image.jpg'.format(dir2save), reals[-1].detach())
        elif opt.train_mode == "harmonization" or opt.train_mode == "editing":
            functions.save_image('{}/{}_wo_mask.jpg'.format(dir2save, _name), reconstruction.detach())
            functions.save_image('{}/real_image.jpg'.format(dir2save), imresize_to_shape(real, reals_shapes[-1][2:], opt).detach())
        return reconstruction

    if scale_w == 1. and scale_h == 1.:
        dir2save_parent = os.path.join(dir2save, "random_samples")
    else:
        reals_shapes = [[r_shape[0], r_shape[1], int(r_shape[2]*scale_h), int(r_shape[3]*scale_w)] for r_shape in reals_shapes]
        dir2save_parent = os.path.join(dir2save, "random_samples_scale_h_{}_scale_w_{}".format(scale_h, scale_w))

    make_dir(dir2save_parent)

    for idx in range(n):
        noise = functions.sample_random_noise(opt.train_stages - 1, reals_shapes, opt)
        sample = netG(noise, reals_shapes, noise_amp)
        functions.save_image('{}/gen_sample_{}.jpg'.format(dir2save_parent, idx), sample.detach())


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--model_dir', help='input image name', required=True)
    parser.add_argument('--gpu', type=int, help='which GPU', default=0)
    parser.add_argument('--num_samples', type=int, help='which GPU', default=50)
    parser.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")
    parser.add_argument('--video_dir', help='input image path', required=True)
    parser.add_argument('--num_images', type=int, default=1)
    parser.add_argument('--vid_ext', default='.jpg', help='ext for video frames')
    parser.add_argument('--out', help = 'save image generate',default='./out/')

    opt = parser.parse_args()
    _gpu = opt.gpu
    _naive_img = opt.naive_img
    __model_dir = opt.model_dir
    opt = functions.load_config(opt)
    opt.gpu = _gpu
    opt.naive_img = _naive_img
    opt.model_dir = __model_dir

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        opt.device = "cuda:{}".format(opt.gpu)

    dir2save = os.path.join(opt.model_dir, "Evaluation")
    make_dir(dir2save)

    fixed_noise_a = []

    print("Loading models...")
    netG_a = torch.load('%s/G_a.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    netG_b = torch.load('%s/G_b.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    fixed_noise_a = torch.load('%s/fixed_noise_a.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    fixed_noise_b = torch.load('%s/fixed_noise_b.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    reals = torch.load('%s/reals_b.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    noise_amp_a = torch.load('%s/noise_amp_a.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    noise_amp_b = torch.load('%s/noise_amp_b.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    reals_shapes = [r.shape for r in reals]

    if opt.train_mode == "generation" or opt.train_mode == "retarget":

        print("Generating Samples...")
        with torch.no_grad():
            # # generate reconstruction
            generate_samples(netG, reals_shapes, noise_amp, reconstruct=True)

            # generate random samples of normal resolution
            rs0 = generate_samples(netG, reals_shapes, noise_amp, n=opt.num_samples)

            # generate random samples of different resolution
            generate_samples(netG, reals_shapes, noise_amp, scale_w=2, scale_h=1, n=opt.num_samples)
            generate_samples(netG, reals_shapes, noise_amp, scale_w=1, scale_h=2, n=opt.num_samples)
            generate_samples(netG, reals_shapes, noise_amp, scale_w=2, scale_h=2, n=opt.num_samples)

    elif opt.train_mode == "harmonization" or opt.train_mode == "editing":
        opt.noise_scaling = 0.1
        _name = "harmonized" if opt.train_mode == "harmonization" else "edited"
        real = functions.read_image_dir(opt.naive_img, opt)
        real = imresize_to_shape(real, reals_shapes[0][2:], opt)
        fixed_noise[0] = real
        if opt.train_mode == "editing":
            fixed_noise[0] = fixed_noise[0] + opt.noise_scaling * \
                                              functions.generate_noise([opt.nc_im, fixed_noise[0].shape[2],
                                                                        fixed_noise[0].shape[3]],
                                                                        device=opt.device)

        out = generate_samples(netG, reals_shapes, noise_amp, reconstruct=True)

        mask_file_name = '{}_mask{}'.format(opt.naive_img[:-4], opt.naive_img[-4:])
        if os.path.exists(mask_file_name):
            mask = functions.read_image_dir(mask_file_name, opt)
            if mask.shape[3] != out.shape[3]:
                mask = imresize_to_shape(mask, [out.shape[2], out.shape[3]], opt)
            mask = functions.dilate_mask(mask, opt)
            out = (1 - mask) * reals[-1] + mask * out
            functions.save_image('{}/{}_w_mask.jpg'.format(dir2save, _name), out.detach())
        else:
            print("Warning: mask {} not found.".format(mask_file_name))
            print("Harmonization/Editing only performed without mask.")

    elif opt.train_mode == "animation":
        print("Generating GIFs...")
        for _start_scale in range(3):
            for _beta in range(80, 100, 5):
                functions.generate_gif(dir2save, netG, fixed_noise, reals, noise_amp, opt,
                                       alpha=0.1, beta=_beta/100.0, start_scale=_start_scale, num_images=100, fps=10)
    
    elif opt.train_mode == "video":
        print("Generating Frame...")
        dataset_a = Video_dataset(opt.video_dir, opt.num_images, opt.vid_ext, opt)
        data_loader_a = DataLoader(dataset_a, shuffle=False ,batch_size=1)
        tmp = (r for r in fixed_noise_a)
       # for a in range(6):
        #    print("Hình dạng của fixed_noise_a: {}".format(tmp.__next__().shape))

        beta = 80
        alpha = 0.1

        with torch.no_grad():
            noise_random  = functions.sample_random_noise(len(fixed_noise_a) - 1, reals_shapes, opt)
            for i in range(len(fixed_noise_a)):
                print("Hình dạng của random noise: {}".format(noise_random[i].shape))

            a = 0
            for data in data_loader_a:  

                ################
                ### Method 1 ###
                ################
                datas = functions.sample_random_noise_video(data, reals_shapes, opt)
                z_curr = [noise_random[i] + datas[i]*noise_amp_b[i] for i in range(len(noise_random))]
                
                mix_g_b = netG_b(z_curr, reals_shapes, noise_amp_b, is_noise = True)
                functions.save_image('{}/b2a_{}.jpg'.format(dir2save,a),mix_g_b.detach())
                
                ################
                ### Method 2 ###
                ################
                '''
                z_prev1 = [0.01 * fixed_noise_a[i] + 0.99 * fakes_a[i] for i in range(len(fixed_noise_b))]
                z_prev2 = fixed_noise_a

                #noise_random = functions.sample_random_noise(len(fixed_noise_b)-1, reals_shapes, opt)
                diff_curr = [beta*(z_prev1[i]-z_prev2[i])+(1-beta)*fakes_a[i] for i in range(len(fixed_noise_b))]
                z_curr = [alpha * fixed_noise_a[i] + (1 - alpha) * (z_prev1[i] + diff_curr[i]) for i in range(len(fixed_noise_b))]
                
                fake_a = netG_a(z_curr,reals_shapes, noise_amp_a)
                functions.save_image('{}/fake_a_{}.jpg'.format(dir2save,a),fake_a.detach())
                
                fake_a = functions.adjust_scales2image(fake_a, opt)
                fakes_a = functions.create_reals_pyramid(fake_a, opt)
                fakes_a = functions.sample_random_noise_video(fakes_a,reals_shapes, opt)

                fakes_a = []
                for i in range(0,len(z_curr)):
                    tmp_a = torch.nn.functional.interpolate(fake_a, 
                                                                  size=[z_curr[i].shape[2],z_curr[i].shape[3]], 
                                                                  mode='bicubic', align_corners=True)
                    fakes_a.append(tmp_a)
                mix_g_b = netG_b(fakes_a, reals_shapes, noise_amp_a, is_noise = True)
                functions.save_image('{}/b2a_{}.jpg'.format(dir2save,a),mix_g_b.detach())
                del fakes_a
                '''
                a = a + 1
            

    print("Done. Results saved at: {}".format(dir2save))

'''
    def generate_gif(dir2save, netG, fixed_noise, reals, noise_amp, opt, alpha=0.1, beta=0.9, start_scale=1,
                 num_images=100, fps=10):
    def denorm_for_gif(img):
        img = denorm(img).detach()
        img = img[0, :, :, :].cpu().numpy()
        img = img.transpose(1, 2, 0) * 255
        img = img.astype(np.uint8)
        return img

    reals_shapes = [r.shape for r in reals]
    all_images = []

    with torch.no_grad():
        noise_random = sample_random_noise(len(fixed_noise) - 1, reals_shapes, opt)
        z_prev1 = [0.99 * fixed_noise[i] + 0.01 * noise_random[i] for i in range(len(fixed_noise))]
        z_prev2 = fixed_noise
        for _ in range(num_images):
            noise_random = sample_random_noise(len(fixed_noise)-1, reals_shapes, opt)
            diff_curr = [beta*(z_prev1[i]-z_prev2[i])+(1-beta)*noise_random[i] for i in range(len(fixed_noise))]
            z_curr = [alpha * fixed_noise[i] + (1 - alpha) * (z_prev1[i] + diff_curr[i]) for i in range(len(fixed_noise))]

            if start_scale > 0:
                z_curr = [fixed_noise[i] for i in range(start_scale)] + [z_curr[i] for i in range(start_scale, len(fixed_noise))]

            z_prev2 = z_prev1
            z_prev1 = z_curr

            sample = netG(z_curr, reals_shapes, noise_amp)
            sample = denorm_for_gif(sample)
            all_images.append(sample)
    imageio.mimsave('{}/start_scale={}_alpha={}_beta={}.gif'.format(dir2save, start_scale, alpha, beta), all_images, fps=fps)
'''	
	
