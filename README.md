# * stable-diffusion-finetuining-pipeline

## * Journey of learning understanding and implementing, also what I have tried:

Stable diffusion was new for me as I have base idea of what it does etc, but in order to optimize it I wanted to dig deep into the network architecture how it is loaded and what can be doen to make it more optimized to finetune on. The jorunery started with exploring the architecture of stable diffusion, once I did then I proceeded with finding how it can be optimized I researched about it and came accross solutions like dreambooth and diffusers and some research papers like dora,etc. I even tried diffusers to see is it really optimized and can it run on the infra requirements we have. 
I wanted to try how optimization can be done from scratch , just like we think layer by layer to really load the model in the gpu infrastructure, The basic idea that came to my mind when I read the doc was  lora, qlora, and knowledge distillation kind of techniques.
So, I created and tried step by step optimizations for each layer and what kind of results do they provide. So I started implementing lora with the network, since the network structure is different lora implemetation also  should target proper layer and network. 

From basics I first understood  how each layer of sdxl contributes to its learning like for instance i understood its layers like this 

1. **VAE (Variational Autoencoder):**
does the job of compressing and decompressing the images,
Compress:  full image and squish it into a smaller, simpler representation called latents.
Decompress: Later, take those latents and turn them back into a full image.

2. **Text Encoders (CLIPTextModel / CLIPTextModelWithProjection):**
   helps in understanding the texxt captions what it really means 
CLIPTextModel → “Just understands the text.”
CLIPTextModelWithProjection → “Understands the text and translates it into a form the image generator (UNet) can directly use.”

4. **UNet (with LoRA):** Does the job of denoise, meaning a randomly noised image in laten space is given ,denoise it iterately to produce somehing that maks sense effectively generating the final image, 90% job happending here.

5. **Noise Scheduler (DDPMScheduler):** Adds and removes the noise for diffusion-based training, meaning it says how much noise to add at each timestep, so the UNet can learn to denoise effectively.


Once I was done with that, I understood unet is the one that needs lora adapter since it does the job of real learning, i.e denoise and predict and learn, and the text encoder also is responsible for learning the style from text because it understands text with text embeddings etc, does the representation of text for image understaing 

So Since I wanted to see how this layers could affect lora in training I went ahead with experiments as follows:

1. First I tried normal pipeline load the networks unet text encoder etc in sub folders of hf or the network basically and put up lora config but here when i tried i was getting nan as my loss reason was vae latent represnetation misatkes in fp16 loading and timesteps which i set initially to zero they were not actually relating the image latent and ocntributing to learning, hence made the modifications for timesteps and vae. so loaded the vae in fp32 did latent operation and converted in fp 16 instead of using vae in 16fp like in dreambotth (i.e community given vae)
2. next the losses were good but I though we could also change the lora adapters for different network like convulation which are essential for unet or cross attnetion importanc=t aspect of unet, so I decided to test out in experiements and find out in inference how the models were getting affected and whther they learnt good or no.
3. Next, I also wanted to try if text encoder as well can help in proper understanding of styles in caption.
4. next i tried their combinations to compare which models are good
5. I came tot hte conclusion that unet with self attention lora performs better so I tried sparse dropout to see in some overfitting kindof aspect can be ocntroleed so tried  spare dropout 
6. Next I wanted to experiment with the dora technique since the technique is much better than plain lora and also assists in resource efficient training, so tried weight decomposiiton dora. 

In short my attempt was to ,
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
    using Weight-Decomposed Low-Rank Adaptation and prodigy optimizer with adam 8 bit optimizers
  ### f. sparse-dropout.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are jus the attention layers with dropout as 0.5 i,.e sparse  
## 2. stabilityai_model_inference.ipynb
     Contains the inference of base model with test prompts

# To run 
* Based on inference or experiements run the according ipynb file 
* The weights of some model which were downloadable from google colab are in google drive 
* Models trained on kaggle didnot allow to download from output folder as well with zipping may be huge so, but i have attached the screenshot of the training models being trained andcheckpoints 


# Links as follows:
https://drive.google.com/drive/folders/1Ie4T_3APD8ALwhKbXn9FDZh5Zi0m5huZ?usp=sharing
have shared only lora as other were on kaggle and didnt let to download i used up google colab resources  2 3 accounts so kaggle 

# Screenshot:
<img width="1322" height="749" alt="image" src="https://github.com/user-attachments/assets/2103b61d-d996-4c3d-99ee-a4ec0532cff2" />

  <hr/>

  
# Experiments idea discussed as follows:

  <hr/>

  
 ### a. text_encoder_yes_unet_many_layers.ipynb 
   #### Detailed High Level overview:
   In this approach what i essentially wanted to load encoder in fp16 vae in fp32 but latent calulation in fp16 for better optimization in term of model loading precision,h ence model could be loaded properly ,Next have lora dapater on text encoder as well as on unet layers, perform vae slicing tiling, perform gradient checkpointing, have noise offset, perform random cropping, dropout and gradientaccumulation these wer all loading, models, memory and trainign optimization, I have explain why each down.
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


### **a. Float16 UNet and Text Encoders**

* less precison 16 than 32 and helps SDXL to fit comfortably within 16GB VRAM, precision lessening helps in VRAM usage reduction

### **b. Float32 VAE for Stability (with FP16 Latents)**

* As said earlier for avoiding nan,  the latents convertsback to FP16 to save memory during UNet training.

### **c. LoRA on Text Encoders & UNet Layers**

* classic importance, trains only small adapter weights instead of the entire model, reduces VRAM usage and compute requirements.

### **d. Gradient Checkpointing**

* Stores fewer activations and recomputes them during backward pass, for VRAM optimization


### **e. VAE Slicing & Tiling**

* Processes images in smaller chunks instead of full resolution at once.
* Prevents VAE encode/decode operations from overflowing VRAM.

### **f. Noise Offset**

* Stabilizes training by smoothing extreme noise values.
* Essentially to reduce the  risks of NaNs or unstable gradients 

### **g. Random Cropping**

* Adds training variety without increasing batch size or resolution.
* Saves memory while improving generalization.

### **h. LoRA Dropout (0.0 for This Run)**

* Keeps LoRA lightweight and avoids creating dense intermediate tensors.
* Minimizes VRAM overhead and stabilizes updates.

### **i. Gradient Accumulation**

* Simulates a larger batch size using multiple micro-steps.
* Enables effective batch training on 16GB VRAM.

   #### Origin of idea (Motivation/Though Process):
  I wanted to see if text encoders can help in understand the text captions better and learn style and unet layers like convulation help in better genrations
   #### Result after trying:
  Not so good.
   #### Why less better/more better (possible reason):
  May be because many layers of unet were trained, meaning adapter for unet work best only on sself attention layers, also text encoder are adding overhead bit not making significant difference.
    
     <hr/>

     
  ### b. text_encoder_yes_unet_single_layer.ipynb
    Here both the text encoder are having lora loaded in quantized manner and the layers of unet are jus the attention layers 
   #### Detailed High Level overview:
   same as above but single unet layer lora
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
  Same as above only here, unet single self attention is updated with lora less update modules for lora, more efficient 
   #### Origin of idea (Motivation/Though Process):
  Same , to try if self attention with text encoder update is better for learning 
   #### Result after trying:
  good outcome for overall anime chacrters but real life characters in anime style not that better like bill gates in naruto style etc
   #### Why less better/more better (possible reason):
   may be text encoder are also updated unet is performing better with single layer based on images comparision but understanding for real person in anime style fails so may be text encoders training may be avoided.

  
    <hr/>

    
  ### c. text_encoder_no_unet_many_layers.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are many 
   #### Detailed High Level overview:
  Same as above only text encoder not training and unet with many layer like ocnvulation self attention etc 
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
  Same as above text encoder no lora only on unet but many layers
   #### Origin of idea (Motivation/Though Process):
  Same as above, based on conclucsion to see if unet layers can give better result with text  encoder freezed completely
   #### Result after trying:
  not good, conclusion, problem lies in training many unet layers, may be not much in encoder
   #### Why less better/more better (possible reason):
   unet layers of self ttention are only better for learning the style properly, may be with text encoder only disavantage then is chacter on real life and overhead of text encoders.

  <hr/>

    ### c. text_encoder_no_unet_single_layers.ipynb
    Here both the text encoder are **not** having lora loaded in quantized manner and the layers of unet are single 
   #### Detailed High Level overview:
  Same as above only text encoder not training and unet with sinle layer 
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
  Same as above text encoder no lora only on unet but single layers more efficient 
   #### Origin of idea (Motivation/Though Process):
  Same as above, based on conclucsion to see if single unet layers can give better result with text  encoder freezed completely
   #### Result after trying:
  much better than other techniques , unet self attention with out text encoders learn nicely without overhead of text encoders
   #### Why less better/more better (possible reason):
   may be because text encoders need more varied captions to learn character real in anime style, rest good for only anime chacretecers with unet single layer, but unet single layer performs good on chacretr real anime an normal anime because maybe those self attention layers need to be trained poeprly i.e lora on them to understand where to focus exactly when image generation i,.e. learn a style in represnetaion 

  <hr/>

   ### e. dora.ipynb
   to try out if dora version of lora is much better than plain lora
   #### Detailed High Level overview:
  Uses dora weight decomposition instead of lora and rest almost same with prodigy optimizers
   ### Tecnhiques used:


* Prodigy optimizer
* DoRA weight decomposition
* Min-SNR gamma loss weighting
* FP16-Fix VAE
* 8-bit Adam (fallback)
* Mixed precision AMP
* Caption dropout
* Fused backward pass
* Advanced noise scheduling
* Gradient checkpointing
* VAE slicing
* VAE tiling (if used in environment)
* Noise offset
* Random cropping
* Gradient accumulation
* FP16 text encoders
* FP16 UNet
* Efficient attention processor (AttnProcessor2_0)
* Frozen VAE & text encoders
* Gaussian-initialized LoRA/DoRA weights


### How Each Technique Helps With Resource-Efficient Training

* **Prodigy optimizer** – Adapts learning automatically, reducing wasted steps and compute.
* **DoRA weight decomposition** – Trains fewer parameters with higher stability and lower memory use.
* **Min-SNR gamma loss weighting** – Prevents high-timestep instability, reducing wasted VRAM-heavy gradients.
* **FP16-Fix VAE** – Cuts 2–4GB VRAM usage while maintaining stable VAE encoding.
* **8-bit Adam (fallback)** – Shrinks optimizer state to save multiple GB of memory.
* **Mixed precision AMP** – Lowers VRAM and speeds up training using half-precision ops.
* **Caption dropout** – Reduces text-encoder compute by skipping full caption embeddings at times.
* **Fused backward pass** – Merges operations to reduce GPU memory overhead and speed up gradients.
* **Advanced noise scheduling** – Lowers unnecessary compute at unstable timesteps.
* **Gradient checkpointing** – Recomputes instead of storing activations to save large amounts of VRAM.
* **VAE slicing** – Processes VAE blocks in slices to cut memory spikes.
* **VAE tiling** – Splits large images into tiles to avoid high-resolution memory blowups.
* **Noise offset** – Stabilizes latents to prevent costly divergence/NaN retries.
* **Random cropping** – Reduces resolution demands by training on smaller regions.
* **Gradient accumulation** – Simulates large batch sizes while keeping per-step VRAM low.
* **FP16 text encoders** – Cuts text-encoder memory in half with minimal quality loss.
* **FP16 UNet** – Reduces UNet compute and VRAM usage significantly.
* **Efficient attention processor (AttnProcessor2_0)** – Lowers attention memory footprint.
* **Frozen VAE & text encoders** – Eliminates backward-pass memory requirements.
* **Gaussian-initialized LoRA/DoRA weights** – Prevents unstable spikes that increase compute cost.



   #### Origin of idea (Motivation/Though Process):
   since DoRA which improves training efficiency by splitting each weight into a direction and a magnitude, allowing the model to learn more useful changes with fewer parameters, makes it more performance good a while not increasing VRAM needs.
   #### Result after trying:
  not that great, unet single layer without text encoders is good 
   #### Why less better/more better (possible reason):
  may be dora may perform good on cross attention layer or so, didnt get time for those experiments may that would be good for dora 
   may be dataset align with lora here more, 1 2 more examperiments with dissrent altercations may help in udnerstanding more. 
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
   
# conclusion:
more optimization techniques and approaches could be tried with experiments or this as well could be extended but this is what I tried for problem statment with more time and research papers and idea more could be tried out
