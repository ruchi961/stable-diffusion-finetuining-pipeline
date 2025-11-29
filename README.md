# stable-diffusion-finetuining-pipeline

Journey and what I have tried:

Stable diffusion was new for me as I have base idea of what it does etc, but in order to optimize it I wanted to dig deep into the network architecture how it is loaded and what can be doen to make it more optimized to finetuin on. The jorunery started with exploring the architecture of stable diffusion, once I did then I proceeded with finding how it can be optimized I researched about it and came accross solutions like dreembooth and diffusers. I even tried diffusers to see is it really optimized and can it run on the infra requirements we have. Even though I did that I wante dto try how optimization can be done from scratch , just like we think layer by layer to really load the model in the gpu infrastructure, The basic idea that came to my ida when optimizing the network was lora, qlora, and knowledge distillation knid of techniques, to understand more of how this basic techniques can be extended I went through some research papers like , where I could laso extend my work to adapt to these tehcniques, but I started with lora the best to train on low VRAM budget 
when I started with lora as i understood the netwrook meaning how each layer of sdxl contributes to its learning like for instance i understood its layers like this 

1. VAE (Variational Autoencoder):
does the job of compressing and decompressing the images,
Compress:  full image and squish it into a smaller, simpler representation called latents.
Decompress: Later, take those latents and turn them back into a full image.
3. Text Encoders (CLIPTextModel / CLIPTextModelWithProjection): helps in understanding the texxt captions what it really means 
CLIPTextModel → “Just understands the text.”
CLIPTextModelWithProjection → “Understands the text and translates it into a form the image generator (UNet) can directly use.”

4. UNet (with LoRA): Does the job of denoise, meaning a randomly noised image in laten space is given ,denoise it iterately to produce somehing that maks sense effectively generating the final image, 90% job happending here.

5. Noise Scheduler (DDPMScheduler): Adds and removes the noise for diffusion-based training, meaning it says how much noise to add at each timestep, so the UNet can learn to denoise effectively.


Once I learned, I also learned unet is the one that needs lora adapter since it does the job of real learning, i.e denoise andpredict and learn, and the text encoder also is responsible for learning the style from text because it understands text anddoes the representation f text to to image understaing 

So Since I wanted to see how this layers qlora affect in training I went ahead with experiments 
one with qlora on unet and text encoder one on unet only etc 
The experiement I tried are below, bu this was my overall idea 
Learn he network architetcure find where the problem lies, how can you optimize, experiment and try how you can do and compare the results.


# Files and folders descriptions:
## 1. Experiments folder 
Contains the performed experiments with the model training and the inference in one ipynb. those with original model such test prompts are only tried.
  ### a. text_encoder_unet_many.ipynb 
    Here both the text encoder are having lora loaded in quantized manner and the layers of unet are many 
  ### b. text_encoder_unet_simple.ipynb
    Here both the text encoder are having lora loaded in quantized manner and the layers of unet are jus the attention layers 
  ### c. text_encoder_no_unet_many.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are many 
  ### d. text_encoder_no_unet_simple.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are simple just the attention layers 
  ### e. dora.ipnb 
    using dora andprodigy optimizer with adam 8 bit optimizers
  ### f. text_encoder_no_unet_simple_sparse.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are jus the attention layers with dropout as 0.5 i,.e sparse  
## 2. inferenceBase.ipynb
     Contains the inference of base model with test prompts

# To run 
* Based on inference or experiements run the according ipynb file 
* The weights of some model which were downloadable from google colab are in google drive 
* Models trained on kaggle didnot allow to download from output folder as well with zipping may be huge so, but i have attached the screenshot of the training models being trained andcheckpoints 


# Links as follows:

# Screenshot:
<img width="1322" height="749" alt="image" src="https://github.com/user-attachments/assets/2103b61d-d996-4c3d-99ee-a4ec0532cff2" />


# Experiments idea discussed as follows:
 ### a. text_encoder_unet_many.ipynb 
   #### Detailed overview:
   #### How it helps:
   #### Origin of idea:
   #### Result after trying:
   #### Why less better/more better (possible reason):
    
    
  ### b. text_encoder_unet_simple.ipynb
    Here both the text encoder are having lora loaded in quantized manner and the layers of unet are jus the attention layers 
   #### Detailed overview:
   #### How it helps:
   #### Origin of idea:
   #### Result after trying:
   #### Why less better/more better (possible reason):
  ### c. text_encoder_no_unet_many.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are many 
   #### Detailed overview:
   #### How it helps:
   #### Origin of idea:
   #### Result after trying:
   #### Why less better/more better (possible reason):
  ### d. text_encoder_no_unet_simple.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are simple just the attention layers 
   #### Detailed overview:
   #### How it helps:
   #### Origin of idea:
   #### Result after trying:
   #### Why less better/more better (possible reason):
  ### e. dora.ipnb 
    using dora andprodigy optimizer with adam 8 bit optimizers
  ### f. text_encoder_no_unet_simple_sparse.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are jus the attention layers with dropout as 0.5 i,.e sparse  
   #### Detailed overview:
   #### How it helps:
   #### Origin of idea:
   #### Result after trying:
   #### Why less better/more better (possible reason):
