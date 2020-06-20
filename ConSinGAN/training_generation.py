import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import ConSinGAN.functions as functions
import ConSinGAN.models as models
import ConSinGAN.imresize as imresize


def train(opt):
    print("Training model with the following parameters:")
    print("\t number of stages: {}".format(opt.train_stages))
    print("\t number of concurrently trained stages: {}".format(opt.train_depth))
    print("\t learning rate scaling: {}".format(opt.lr_scale))
    print("\t non-linearity: {}".format(opt.activation))

    real_a = functions.read_image_a(opt)
    real_b = functions.read_image_b(opt)

    real_b =functions.torch2uint8(real_b)
    real_b =imresize.imresize_in(real_b, output_shape=(real_a.shape[2],real_a.shape[3]))
    #real_b_1 = functions.np2torch(real_b,opt)
    real_b = functions.np2torch(real_b,opt)
    #print(real_b_1.shape)
    
    real_a = functions.adjust_scales2image(real_a, opt) #Caculate size image of first scale
    reals_a = functions.create_reals_pyramid(real_a, opt)

    real_b = functions.adjust_scales2image(real_b, opt)
    reals_b = functions.create_reals_pyramid(real_b, opt)
    
    #Cáº§n lÃ m cho kÃ­ch thÆ°á»›c cá»§a reals_a vÃ  reals_b nÃ³ báº±ng nhau

    print("Training on image pyramid: {}".format([r.shape for r in reals_a]))
    print("Training on image pyramid: {}".format([r.shape for r in reals_b]))

    print("")

    #Add generator A
    generator_a = init_G(opt)
    fixed_noise_a = []
    noise_amp_a = []
    fakes_a = []
    mixs_g_a =[]

    #Add generator B
    generator_b = init_G(opt)
    fixed_noise_b = []
    noise_amp_b = []
    fakes_b =[]
    mixs_g_b =[]

    for scale_num in range(opt.stop_scale+1):
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                print(OSError)
                pass
        functions.save_image('{}/real_a_scale.jpg'.format(opt.outf), reals_a[scale_num])
        functions.save_image('{}/real_b_scale.jpg'.format(opt.outf), reals_b[scale_num])
        #functions.save_image('{}/real_b_test.jpg'.format(opt.outf), real_b_1)


        d_curr_a = init_D(opt)
        d_curr_b = init_D(opt)
        if scale_num > 0: #Neu nhu da train dc 1 lan
            d_curr_a.load_state_dict(torch.load('%s/%d/netD_a.pth' % (opt.out_,scale_num-1)))
            d_curr_b.load_state_dict(torch.load('%s/%d/netD_b.pth' % (opt.out_,scale_num-1)))
            generator_a.init_next_stage()
            generator_b.init_next_stage()

        #writer = SummaryWriter(log_dir=opt.outf)
        fixed_noise_a, fixed_noise_b, noise_amp_a,noise_amp_b, generator_a, generator_b, d_curr_a, d_curr_b, fakes_a, fakes_b, mixs_g_a, mixs_g_b = train_single_scale(d_curr_a, d_curr_b, 
                                                              generator_a, generator_b, reals_a, reals_b, 
                                                              fixed_noise_a, fixed_noise_b, 
                                                              noise_amp_a,noise_amp_b, 
                                                              fakes_a, fakes_b, 
                                                              mixs_g_a,mixs_g_b,
                                                              opt, scale_num)
        model_summary(generator_a)
        #print("Chiều dài fakes a là: ", len(fakes_a))
        #print("Chiều dài fakes b là: ",len(fakes_b))
        torch.save(fixed_noise_a, '%s/fixed_noise_a.pth' % (opt.out_))
        torch.save(fixed_noise_b, '%s/fixed_noise_b.pth' % (opt.out_))
        torch.save(generator_a, '%s/G_a.pth' % (opt.out_))
        torch.save(generator_b, '%s/G_b.pth' % (opt.out_))
        torch.save(reals_a, '%s/reals_a.pth' % (opt.out_))
        torch.save(reals_b, '%s/reals_b.pth' % (opt.out_))
        torch.save(noise_amp_a, '%s/noise_amp_a.pth' % (opt.out_))
        torch.save(noise_amp_b, '%s/noise_amp_b.pth' % (opt.out_))
        del d_curr_a, d_curr_b
    #writer.close()
    return


def train_single_scale(netD_a, netD_b, netG_a, netG_b, reals_a, reals_b, fixed_noise_a,fixed_noise_b, noise_amp_a,noise_amp_b, fakes_a,fakes_b, mixs_g_a, mixs_g_b, opt, depth):
    reals_shapes = [real.shape for real in reals_a]
    real_a = reals_a[depth]
    real_b = reals_b[depth]


    # Hai sieu tham so cua loss rec vÃ  cycle
    alpha = opt.alpha
    beta = opt.beta

    ############################
    # define z_opt for training on reconstruction
    ###########################
    if depth == 0:
        if opt.train_mode == "generation" or opt.train_mode == "retarget":
            z_opt_a = reals_a[0]
            z_opt_b = reals_b[0]
        elif opt.train_mode == "animation":
            z_opt = functions.generate_noise([opt.nc_im, reals_shapes[depth][2], reals_shapes[depth][3]],
                                             device=opt.device).detach()
    else:
        if opt.train_mode == "generation" or opt.train_mode == "animation":
            z_opt_a = functions.generate_noise([opt.nfc,
                                              reals_shapes[depth][2]+opt.num_layer*2,
                                              reals_shapes[depth][3]+opt.num_layer*2],
                                              device=opt.device)
            z_opt_b = functions.generate_noise([opt.nfc,
                                              reals_shapes[depth][2]+opt.num_layer*2,
                                              reals_shapes[depth][3]+opt.num_layer*2],
                                              device=opt.device)                                  
        else:
            z_opt = functions.generate_noise([opt.nfc, reals_shapes[depth][2], reals_shapes[depth][3]],
                                              device=opt.device).detach()
    fixed_noise_a.append(z_opt_a.detach())
    fixed_noise_b.append(z_opt_b.detach())

    ############################
    # define optimizers, learning rate schedulers, and learning rates for lower stages
    ###########################
    # setup optimizers for D
    optimizerD = optim.Adam(list(netD_a.parameters()) + list(netD_b.parameters()), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    
    # setup optimizers for G
    # remove gradients from stages that are not trained
    for block in netG_a.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    for block in netG_b.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list_a = [{"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale**(len(netG_a.body[-opt.train_depth:])-1-idx))}
               for idx, block in enumerate(netG_a.body[-opt.train_depth:])]

    parameter_list_b = [{"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale**(len(netG_b.body[-opt.train_depth:])-1-idx))}
               for idx, block in enumerate(netG_b.body[-opt.train_depth:])]


    # add parameters of head and tail to training
    if depth - opt.train_depth < 0:
        parameter_list_a += [{"params": netG_a.head.parameters(), "lr": opt.lr_g * (opt.lr_scale**depth)}]
        parameter_list_b += [{"params": netG_b.head.parameters(), "lr": opt.lr_g * (opt.lr_scale**depth)}]
    parameter_list_a += [{"params": netG_a.tail.parameters(), "lr": opt.lr_g}]
    parameter_list_b += [{"params": netG_b.tail.parameters(), "lr": opt.lr_g}]
    optimizerG = optim.Adam(parameter_list_a + parameter_list_b, lr=opt.lr_g, betas=(opt.beta1, 0.999))
    #optimizerG_b = optim.Adam(parameter_list_b, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # define learning rate schedules
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8*opt.niter], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8*opt.niter], gamma=opt.gamma)

    ############################
    # calculate noise_amp
    ###########################
    if depth == 0:
        noise_amp_a.append(1)
        noise_amp_b.append(1)
    else:
        criterion = nn.MSELoss()

        # noise_amp_a
        noise_amp_a.append(0)
        z_reconstruction_a = netG_a(fixed_noise_a, reals_shapes, noise_amp_a)      

        rec_loss_a = criterion(z_reconstruction_a, real_a)
        RMSE_a = torch.sqrt(rec_loss_a).detach()
        _noise_amp_a = opt.noise_amp_init_a * RMSE_a
        noise_amp_a[-1] = _noise_amp_a

        # noise_amp_b
        noise_amp_b.append(0)
        z_reconstruction_b = netG_b(fixed_noise_b, reals_shapes, noise_amp_b)

        rec_loss_b = criterion(z_reconstruction_b, real_b)
        RMSE_b = torch.sqrt(rec_loss_b).detach()
        _noise_amp_b = opt.noise_amp_init_b * RMSE_b
        noise_amp_b[-1] = _noise_amp_b
        
    # start training
    _iter = tqdm(range(opt.niter))
    for iter in _iter:
        _iter.set_description('stage [{}/{}]:'.format(depth, opt.stop_scale))

        ############################
        # (0) sample noise for unconditional generation
        ###########################
        noise = functions.sample_random_noise(depth, reals_shapes, opt)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            #############################
            ####      Train D_a      ####
            #############################

            netD_a.zero_grad()
            
            # train with real
            output = netD_a(real_a)
            errD_real = -2*output.mean()

            # train with fake
            if j == opt.Dsteps - 1:
                fake_a = netG_a(noise, reals_shapes, noise_amp_a)
            else:
                with torch.no_grad():
                    fake_a = netG_a(noise, reals_shapes, noise_amp_a)
            
            
            #Viết hàm để tránh trường hợp add vào quá nhiều tạo ra tràn bộ nhớ cho fakes a
            if(len(fakes_a) == depth):
                fakes_a.append(fake_a)
            else:
                fakes_a.pop(depth)
                fakes_a.append(fake_a)

            if (fakes_a[-1].shape[2] != noise[-1].shape[2] or fakes_a[-1].shape[3] != noise[-1].shape[3]):
                fakes_a[-1] = torch.nn.functional.interpolate(fakes_a[-1], size=[noise[-1].shape[2],noise[-1].shape[3]], mode='bicubic', align_corners=True)
                     
            

            if j == opt.Dsteps - 1:
                fake_b = netG_b(noise, reals_shapes, noise_amp_b)
            else:
                with torch.no_grad():
                    fake_b = netG_b(noise, reals_shapes, noise_amp_b)


            #Viết hàm để tránh trường hợp add vào quá nhiều tạo ra tràn bộ nhớ cho fakes a
            if(len(fakes_b) == depth):
                fakes_b.append(fake_b)
            else:
                fakes_b.pop(depth)
                fakes_b.append(fake_b)

            if (fakes_b[-1].shape[2] != noise[-1].shape[2] or fakes_b[-1].shape[3] != noise[-1].shape[3]):
                fakes_b[-1] = torch.nn.functional.interpolate(fakes_b[-1], size=[noise[-1].shape[2],noise[-1].shape[3]], mode='bicubic', align_corners=True)

            if(depth !=0):
                mix_g_a = netG_a(fakes_b, reals_shapes,noise_amp_b, is_noise = True)
            else:
                mix_g_a = netG_a(fakes_b, reals_shapes,noise_amp_b)

            #mix_g_a = netG_a(fakes_b, reals_shapes,noise_amp_b)
            #mixs_g_a.append(mix_g_a)
            output_a = netD_a(mix_g_a.detach())
            output_a2 = netD_a(fake_a.detach())
            errD_fake_a = output_a.mean() + output_a2.mean()

            gradient_penalty_a = functions.calc_gradient_penalty(netD_a, real_a, mix_g_a, opt.lambda_grad, opt.device)
            gradient_penalty_a += functions.calc_gradient_penalty(netD_a, real_a, fake_a, opt.lambda_grad, opt.device)
            errD_total_a = errD_real + errD_fake_a + gradient_penalty_a
            errD_total_a.backward(retain_graph=True)
            
            # Tại mỗi scale thì chèn vào mix duy nhất
            if(len(mixs_g_a) == depth):
                mixs_g_a.append(mix_g_a)
            else:
                mixs_g_a.pop(depth)
                mixs_g_a.append(mix_g_a)

            if (mixs_g_a[-1].shape[2] != noise[-1].shape[2] or mixs_g_a[-1].shape[3] != noise[-1].shape[3]):
                mixs_g_a[-1] = torch.nn.functional.interpolate(mixs_g_a[-1], size=[noise[-1].shape[2],noise[-1].shape[3]], mode='bicubic', align_corners=True)

            #############################
            ####      Train D_b      ####
            #############################
            netD_b.zero_grad()

            # train with real
            output = netD_b(real_b)
            errD_real = -2*output.mean()

            # train with fake
            if(depth !=0):
                mix_g_b = netG_b(fakes_a, reals_shapes, noise_amp_a,is_noise = True)
            else:
                mix_g_b = netG_b(fakes_a, reals_shapes, noise_amp_a)

            
            #mixs_g_b.append(mix_g_b)
            output_b = netD_b(mix_g_b.detach()) 
            output_b2 = netD_b(fake_b.detach())
            errD_fake_b = output_b.mean() + output_b2.mean()

            gradient_penalty_b = functions.calc_gradient_penalty(netD_b, real_b, fake_b, opt.lambda_grad, opt.device)
            gradient_penalty_b += functions.calc_gradient_penalty(netD_b, real_b, mix_g_b, opt.lambda_grad, opt.device)
            errD_total_b = errD_real + errD_fake_b + gradient_penalty_b
            errD_total_b.backward(retain_graph=True)

            # Tại mỗi scale thì chèn vào mix duy nhất
            if(len(mixs_g_b) == depth):
                mixs_g_b.append(mix_g_b)
            else:
                mixs_g_b.pop(depth)
                mixs_g_b.append(mix_g_b)
            
            if (mixs_g_b[-1].shape[2] != noise[-1].shape[2] or mixs_g_b[-1].shape[3] != noise[-1].shape[3]):
                mixs_g_b[-1] = torch.nn.functional.interpolate(mixs_g_b[-1], size=[noise[-1].shape[2],noise[-1].shape[3]], mode='bicubic', align_corners=True)

            optimizerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        netG_a.zero_grad()
        netG_b.zero_grad()
        
        output_a = netD_a(mix_g_a)
        output_a2 = netD_a(fake_a)
        errG_a = -output_a.mean() - output_a2.mean()

        output_b = netD_b(mix_g_b)
        output_b2 = netD_b(fake_b)
        errG_b = (-output_b.mean() - output_b2.mean())

        #print(len(noise_amp_a))

        if alpha != 0:
            rec_loss = nn.MSELoss()
            rec_a = netG_a(fixed_noise_a, reals_shapes, noise_amp_a) #xem lai
            rec_loss_a = alpha * rec_loss(rec_a, real_a)

            rec_b = netG_b(fixed_noise_b, reals_shapes, noise_amp_b)
            rec_loss_b = alpha * rec_loss(rec_b, real_b)
        else:
            rec_loss_a = 0
            rec_loss_b = 0

        if beta != 0:
            cycle_loss = nn.MSELoss()
            
            if (depth != 0):
                cycle_a = netG_a(mixs_g_b,reals_shapes, noise_amp_a, is_noise =True) #if else chỗ này tai scale thứ 2 trở đi tăng chanel lên 64 
                cycle_b = netG_b(mixs_g_a,reals_shapes, noise_amp_b, is_noise = True)
            else:
                cycle_a = netG_a(mixs_g_b,reals_shapes, noise_amp_a) #if else chỗ này tai scale thứ 2 trở đi tăng chanel lên 64 
                cycle_b = netG_b(mixs_g_a,reals_shapes, noise_amp_b)        
            
            cycle_loss_a = beta * cycle_loss(cycle_a, fake_a)
            cycle_loss_b = beta * cycle_loss(cycle_b, fake_b)
        else:
            cycle_loss_a = 0
            cycle_loss_b = 0
                
        errG_total_a = errG_a + rec_loss_a + cycle_loss_a
        errG_total_a.backward(retain_graph=True)

        errG_total_b = errG_b + rec_loss_b + cycle_loss_b
        errG_total_b.backward(retain_graph=True)

        for _ in range(opt.Gsteps):
            optimizerG.step()

        ############################
        ####  (3) Log Results   ####
        ############################
        #if iter % 250 == 0 or iter+1 == opt.niter:
        #    print(f"[{iter}]/20000 tại scale [depth]")
            #writer.add_scalar('Loss/train/D_a/real/{}'.format(j), -errD_real.item(), iter+1)
            #writer.add_scalar('Loss/train/D_a/fake/{}'.format(j), errD_fake.item(), iter+1)
            #writer.add_scalar('Loss/train/D/gradient_penalty/{}'.format(j), gradient_penalty.item(), iter+1)
            #writer.add_scalar('Loss/train/G/gen', errG.item(), iter+1)
            #writer.add_scalar('Loss/train/G/reconstruction', rec_loss.item(), iter+1)
        if iter % 500 == 0 or iter+1 == opt.niter:
            functions.save_image('{}/fake_sample_a{}.jpg'.format(opt.outf, iter+1), fake_a.detach())
            functions.save_image('{}/reconstruction_a{}.jpg'.format(opt.outf, iter+1), rec_a.detach())
            functions.save_image('{}/fake_sample_b{}.jpg'.format(opt.outf, iter+1), fake_b.detach())
            functions.save_image('{}/reconstruction_b{}.jpg'.format(opt.outf, iter+1), rec_b.detach())
            functions.save_image('{}/b2a_{}.jpg'.format(opt.outf,iter+1),mix_g_a.detach())
            functions.save_image('{}/a2b_{}.jpg'.format(opt.outf,iter+1),mix_g_b.detach())
           # generate_samples(netG_a, opt, depth, noise_amp, writer, reals, iter+1)

        schedulerD.step()
        schedulerG.step()
        # break
    print("Fakes_a: {}".format([r.shape for r in fakes_a]))
    print("Fakes_b: {}".format([r.shape for r in fakes_b]))
    functions.save_networks(netG_a,netG_b, netD_a,netD_b, z_opt_a,z_opt_b, opt)
    return fixed_noise_a, fixed_noise_b, noise_amp_a, noise_amp_b, netG_a, netG_b, netD_a, netD_b, fakes_a,fakes_b, mixs_g_a, mixs_g_b


def generate_samples(netG, opt, depth, noise_amp, writer, reals, iter, n=25):
    opt.out_ = functions.generate_dir2save(opt)
    dir2save = '{}/gen_samples_stage_{}'.format(opt.out_, depth)
    reals_shapes = [r.shape for r in reals]
    all_images = []
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    with torch.no_grad():
        for idx in range(n):
            noise = functions.sample_random_noise(depth, reals_shapes, opt)
            sample = netG(noise, reals_shapes, noise_amp)
            all_images.append(sample)
            functions.save_image('{}/gen_sample_{}.jpg'.format(dir2save, idx), sample.detach())

        all_images = torch.cat(all_images, 0)
        all_images[0] = reals[depth].squeeze()
        grid = make_grid(all_images, nrow=min(5, n), normalize=True)
        writer.add_image('gen_images_{}'.format(depth), grid, iter)


def init_G(opt):
    # generator initialization:
    netG = models.GrowingGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)
    print(netG)
    # print(netG)

    return netG

def init_D(opt):
    #discriminator initialization:
    netD = models.Discriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    
    # print(netD)

    return netD


def model_summary(model):
  print("model_summary")
  print()
  print("Layer_name"+"\t"*7+"Number of Parameters")
  print("="*100)
  model_parameters = [layer for layer in model.parameters() if layer.requires_grad] #Parameter training
  layer_name = [child for child in model.children()]
  j = 0
  total_params = 0
  print("\t"*10)
  for i in layer_name:
    print()
    param = 0
    try:
      bias = (i.bias is not None)
    except:
      bias = False  
    if not bias:
      param =model_parameters[j].numel()+model_parameters[j+1].numel()
      j = j+2
    else:
      param =model_parameters[j].numel()
      j = j+1
    print(str(i)+"\t"*3+str(param))
    total_params+=param
  print("="*100)
  print(f"Total Params:{total_params}")       

