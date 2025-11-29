# * stable-diffusion-finetuining-pipeline

## * Journey and what I have tried:

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
  ### a. text_encoder_yes_unet_many_layer.ipynb 
    Here both the text encoder are having lora loaded in quantized manner and the layers of unet are many 
  ### b. text_encoder_yes_unet_single_layer.ipynb
    Here both the text encoder are having lora loaded in quantized manner and the layers of unet are jus the attention layers 
  ### c. text_encoder_no_unet_many_layer.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are many 
  ### d. text_encoder_no_unet_single_layer.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are simple just the attention layers 
  ### e. dora.ipnb 
    using dora andprodigy optimizer with adam 8 bit optimizers
  ### f. sparse-dropout.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are jus the attention layers with dropout as 0.5 i,.e sparse  
## 2. stabilityai_model_inference.ipynb
     Contains the inference of base model with test prompts

# To run 
* Based on inference or experiements run the according ipynb file 
* The weights of some model which were downloadable from google colab are in google drive 
* Models trained on kaggle didnot allow to download from output folder as well with zipping may be huge so, but i have attached the screenshot of the training models being trained andcheckpoints 


# Links as follows:

# Screenshot:
<img width="1322" height="749" alt="image" src="https://github.com/user-attachments/assets/2103b61d-d996-4c3d-99ee-a4ec0532cff2" />


# Experiments idea discussed as follows:
 ### a. text_encoder_yes_unet_many_layers.ipynb 
   #### Detailed High Level overview:
   ### Tecnhiques used:
   
  a. Float16 UNet and text encoders

  
  b. Float32 VAE for stability -> later conversion in latent space for 16bit and back to 32 bit vae this was done because vae 16bit loaded produces nan errors 

  
  c. LoRA on text encoders and unet many layers like convolutions, cross attention

  
  d. Gradient checkpointing
  
  e. VAE slicing and tiling
  
  f. Noise offset
  
  g. Random cropping for more training variety

  
  h. 0.0 dropout through LoRA dropout

  
  i. Gradient accumulation to simulate larger batch size
  

   #### How it helps for Resource-Efficient Model Training such as 16 GB VRAM:
   #### Origin of idea (Motivation/Though Process):
   #### Result after trying:
   #### Why less better/more better (possible reason):
    
   
  ### b. text_encoder_yes_unet_single_layer.ipynb
    Here both the text encoder are having lora loaded in quantized manner and the layers of unet are jus the attention layers 
   #### Detailed High Level overview:
   ### Tecnhiques used:
  a. Float16 UNet and text encoders

  
  b. Float32 VAE for stability -> later conversion in latent space for 16bit and back to 32 bit vae this was done because vae 16bit loaded produces nan errors 

  
  c. LoRA on text encoders and unet single self attention layer
  d. Gradient checkpointing
  e. VAE slicing and tiling
  f. Noise offset
  g. Random cropping for more training variety
  h. 0.0 dropout through LoRA dropout
  i. Gradient accumulation to simulate larger batch size
   #### How it helps for Resource-Efficient Model Training such as 16 GB VRAM:
   #### Origin of idea (Motivation/Though Process):
   #### Result after trying:
   #### Why less better/more better (possible reason):
   
  ### c. text_encoder_no_unet_many_layers.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are many 
   #### Detailed High Level overview:
   ### Tecnhiques used:
   a. Float16 UNet and text encoders
  b. Float32 VAE for stability -> later conversion in latent space for 16bit and back to 32 bit vae this was done because vae 16bit loaded produces nan errors 
  c. LoRA unet many layers
  d. Gradient checkpointing
  e. VAE slicing and tiling
  f. Noise offset
  g. Random cropping for more training variety
  h. 0.0 dropout through LoRA dropout
  i. Gradient accumulation to simulate larger batch size
   #### How it helps for Resource-Efficient Model Training such as 16 GB VRAM:
   #### Origin of idea (Motivation/Though Process):
   #### Result after trying:
   #### Why less better/more better (possible reason):
   
  ### d. text_encoder_no_unet_single_layer.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are simple just the attention layers 
   #### Detailed High Level overview:
   ### Tecnhiques used:
   
  a.  Float16 UNet and text encoders

  
  b.  Float32 VAE for stability -> later conversion in latent space for 16bit and back to 32 bit vae this was done because vae 16bit loaded produces nan errors 

  
  c.  LoRA on text encoders and unet single self attention layer

  
  d.  Gradient checkpointing

  
  e.  VAE slicing and tiling

  
  f. Noise offset

  
  g. Random cropping for more training variety

  
  h. 0.0 dropout through LoRA dropout

  
  i. Gradient accumulation to simulate larger batch size

  
   #### How it helps for Resource-Efficient Model Training such as 16 GB VRAM:
   #### Origin of idea (Motivation/Though Process):
   #### Result after trying:
   #### Why less better/more better (possible reason):
   
    using dora andprodigy optimizer with adam 8 bit optimizers

  <hr/>

  
  ### f. sparse_dropout.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are jus the attention layers with dropout as 0.5 i,.e sparse  
  #### Detailed High Level overview:
    same abbove techniques , It freezes large components, applies LoRA only to the UNet, and uses memory-saving tricks like gradient checkpointing, slicing, and float16 models to fit SDXL training into limited VRAM, but here specially makes use of sparse dropout 0.5, 
   ### Tecnhiques used:
  a. same above features only 

  
  b. unet with attention layer lora 

  
  c. sparse-style dropout through LoRA dropout

  

   #### How it helps for Resource-Efficient Model Training such as 16 GB VRAM:
   Sparse dropout reduces the number of active LoRA parameters during training, lowering memory usage (VRAM SAVING) and preventing the model from overfitting to dense feature patterns. 
   
   #### Origin of idea (Motivation/Though Process):
   I wanted to try out if results were better since it is sparse dropout means half of the lora updates are disabled this was just to see if lessens overfitting meaning the apt learning for this I though the learning in other methods was maybe even though not overfitting in terms of loss directly bbut in style or image difference was bit overfiiting r lossing the style or not properly adapting the style, so wanted to try sparse drop out to see if results were better.
   #### Result after trying: 
   Results were better than other in terms of image generation and style understanding espicaially for naruto bill gates and other reaal life characters in naruto style representations.
   #### Why less better/more better (possible reason):
   may be because of dropout 0.5 than 0 on unet attention layers it was able to learn properly, especially cases where style was to be adpated for human being personalities 
   
