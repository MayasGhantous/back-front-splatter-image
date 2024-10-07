# back-front-splatter-image
We forked this code from the official implementation of **"Splatter Image: Ultra-Fast Single-View 3D Reconstruction"** and then implemented our approach on top of the original one, you can find the orgianl code [here](https://github.com/szymanowiczs/splatter-image/tree/main).



## Our Approach
Instead of creating one splatter for the object, we create two: one for the back Gaussians and one for the front Gaussians. How do we do this? By creating a depth loss, which implies that the back Gaussians must be farther than the front Gaussians.
currently our code works just in the car dataset, and how to trian a model and use it you can see the orgianl code.

# Results
we have 2 steps in training as it appears in the orgianl code, regular training were we have just l2 loss and depth-loss
and the second step fine-tunning which also has lpips-loss

##### Losses before fine-tuning

<p float="left">
  <img src="https://github.com/user-attachments/assets/66118e83-78bd-4d64-abe9-7e8f8a3c4d65" width="300" />
  <img src="https://github.com/user-attachments/assets/9d32dd74-3240-48d5-baa5-20b3f1f4a9da" width="300" />
  <img src="https://github.com/user-attachments/assets/8ee36365-b5ca-4811-bbb5-6b2bc5304830" width="300" />
</p>


##### Losses after fine-tuning

<p float="left">
  <img src="https://github.com/user-attachments/assets/02cd553e-36ab-4db6-aa84-65f9648bc78e" width="300" />
  <img src="https://github.com/user-attachments/assets/f0ce91c9-2fb2-4aa3-a995-f47255a9c46d" width="300" />
  <img src="https://github.com/user-attachments/assets/96b740a2-9726-43e8-a135-0098b2dea650" width="300" />
  <img src="https://github.com/user-attachments/assets/e3efffc3-71ec-4b64-8284-6791b8dc364a" width="300" />
</p>


### Comparison of Results at 40k Iterations and 30k Finetuning

| Metric          | Original Results | Our Results   |
|-----------------|------------------|---------------|
| **PSNR value**   | 22.1883          | 22.41871      |
| **LPIPS value**  | 0.15895          | 0.15271       |
| **SSIM value**   | 0.88761          | 0.89254       |



# Validation Test Results

### Our Result:
![Our Result](https://github.com/user-attachments/assets/7e978f18-60de-4088-a306-90e89a91e5da)

### Original Result:
![output_video (15)](https://github.com/user-attachments/assets/0f16d21d-4b7a-4737-b43a-2947f46031a2)

### Ground Truth:
![Ground Truth](https://github.com/user-attachments/assets/ce935deb-7552-4a66-8d3e-828ff27a50f6)




### For more details, read this presentation:

[View Presentation](https://docs.google.com/presentation/d/1w29UsamSIe1vavGOS0ox0E60W_vwTnMA/edit?usp=sharing&ouid=104114095327066430401&rtpof=true&sd=true)

# Acknowledgements







