
# Advanced Deep Learning Models in Earth Observation for Urban Applications: Bologna and Turin Case Studies


**Abstract**

This project investigates the application of deep learning models for building segmentation and footprint extraction using pansharpened satellite imagery, with the aim of supporting or optimizing urban data management and enhancing environmental resource management. Building footprint extraction is critical for identifying the exact spatial extent of structures, which can be used to inform various urban planning and resource management efforts. The study utilized panchromatic and multispectral satellite images from the WorldView-2 platform to assess multiple deep learning architectures, including UNet with attention mechanisms, DHAUNet with hybrid and multi-head attention configurations, and several iterations of DEEPLabV3 with ResNet backbones. Among these, DEEPLabV3 with ResNet-50, tested across different tile sizes (1024, 512, and 256), demonstrated the highest accuracy in extracting building footprints, with optimal results achieved using a 256×256 tile size. The refined model was applied to datasets from two cities - Turin and Bologna - representing distinct urban roof architectures. Additionally, a combined dataset facilitated the comparative analysis of segmentation performance across varying urban environments. The results emphasize the model’s adaptability and precision in extracting building footprints, demonstrating its robustness across diverse architectural contexts. This research contributes to the advancement of deep learning integration with satellite remote sensing, providing valuable insights for future applications in sustainable urban development. Furthermore, the findings underscore the potential for leveraging accurate building footprint data to support renewable energy initiatives, disaster risk management, and improved urban planning practices globally.


**What’s the Aim of This Thesis?**
This thesis focuses on using deep learning to extract building footprints, crucial for urban planning and infrastructure. This method reduces costs and labor, especially in cities lacking detailed surveys. It also supports resource management, renewable energy planning, and disaster risk management in areas with limited data. 



the dataset used for deep learning comes from WorldView-2 imagery for both Bologna and Turin. We use multispectral images with a 1.85-meter resolution for land cover analysis, and panchromatic images at 0.46 meters to capture finer details. By combining both, we create pansharpened images, which are high-resolution and rich in detail. At the bottom of the slide, you can see vector data from the local governments of Bologna and Turin. These provide precise building footprints, which we use to validate our deep learning models for accuracy. 

![image](https://github.com/user-attachments/assets/510079b5-3d59-45cf-8a9b-9054da8a1f5e)

This flowchart shows the key steps in developing deep learning models for image segmentation. First, we prepare the dataset, which includes images and their corresponding vectors, and split it into training, validation, and test datasets. The model is built and trained using the training dataset, while the validation dataset helps adjust the model’s parameters. After training, we evaluate its performance using the testing dataset and metrics like accuracy, mean square error, and others.

![image](https://github.com/user-attachments/assets/2277db47-e0c1-4fb4-8e0f-9f2dcee81251)

I used three types of architectures: UNet, Dual Hybrid Attention UNet, and DeepLabV3. 
•	UNet works in two steps: first, the encoder reduces the image size while extracting important features like edges and shapes. Then, the decoder restores the image to its original size, identifying objects like buildings or roads. 
•	Dual Hybrid Attention is similar to UNet but works differently: it first focuses on key areas in the image, and then Channel-wise attention assigns different importance to each color channel (Red, Green, and Blue), making it more precise for complex city images.
•	DeepLabV3 is an advanced model that analyzes both small details and the overall structure of an image simultaneously. It uses a method to examine the image at different scales, allowing it to capture key features like edges, textures, and shapes more effectively, resulting in more accurate object labeling.
I trained 25 models with varying hyperparameters and pixel sizes to optimize segmentation performance. The dataset used comprised around 15,000 images and corresponding vector labels for training and testing each model. Each model was trained for 100 iterations. Depending on the complexity and hyperparameters, one iteration took between 4 to 15 minutes. Training on my 8 GB GPU laptop took approximately 25 hours per model. However, using Google Colab with a 40 GB GPU reduced training time between 2 and 8 hours per model. Designing and training the models took over 100 hours for UNet and DHAUnet, and more than 200 hours for the DeepLabV3 architecture. In total, the process spanned approximately four months. 

![image](https://github.com/user-attachments/assets/df52cf3c-5e2c-4a9e-9fa4-7c83a263b96f)

I used several key metrics to evaluate the performance of deep learning models. Accuracy measured the proportion of correctly classified pixels, while the Dice Coefficient and (IoU) quantified the overlap between predicted and actual segmentation masks. Mean Squared Error assessed the average squared difference between predictions and true values. Precision measured the proportion of correct positive predictions, while Recall indicated how many actual positives were correctly identified. The F1-Score balanced these two.
The figures presented here show a sample result.
•	True Positive (Green): Areas where the prediction matches the ground truth.
•	False Positive (Red): Over-predictions by the model.
•	False Negative (Blue): Missed predictions where the model failed to predict existing footprints.

![image](https://github.com/user-attachments/assets/683e90c0-0982-4f0a-beb5-ada652ad9c68)

The table you're looking at presents the results of 15 models. These models are based on different configurations, including deep learning architectures, datasets (Turin, Bologna, and a combined dataset), and pixel sizes. In this table, you can see two models: First Model and Second Model. Both use the DeepLabV3 architecture, but the difference between them lies in their hyperparameters during the learning process. Among the models, Number 14 (Second Model, trained on the Bologna dataset with a 256×256-pixel size) achieved the best performance, with an accuracy of 93.5% and the mean square error of 0.05. The images shown here are samples from Bologna with the best model's vector overlaying the building footprints. 

![image](https://github.com/user-attachments/assets/9a96dd71-2a4d-4e6d-b5bc-47f2e6a64403)

I have used my extracted building footprints for two case studies, and Application #1 focuses on Turin’s Solar Energy Potential Assessment. However, implementing this in reality faces limitations, including legal restrictions in historical areas and the absence of Lidar data, which affects roof orientation accuracy. Despite these challenges, we analyzed a 65 km² portion of Turin, covering over 11,600 buildings with a combined rooftop area of around 230,000 m². I also filtered down the roof areas between 100 and 10,000 square meters. Using deep learning models and solar radiation data obtained from the Solcast website , we estimate an annual electricity generation potential of roughly 121.5 GWh.

![image](https://github.com/user-attachments/assets/3cb3e7f9-9055-409f-b678-3d044dbacc0a)

Application #2 focuses on Bologna's rainwater harvesting potential. Despite challenges like heritage laws and drainage system issues affecting prediction accuracy, we analyzed a twelve square kilometer area. Starting with over thirty-six hundred rooftops, we filtered down to more than thirty-three hundred buildings with rooftops over 50 square meter covering around 3.45 million square meters. Combining local rainfall data from 2001 to 2023, provided by the Commune di Bologna, with roof material data thanks to Francesca and Professor Bitelli, and using runoff coefficients, the model predicts that over 1.31 million cubic meters of rainwater could be harvested in 2030. 

![image](https://github.com/user-attachments/assets/55accc32-7528-4bd3-a5a1-506244657dbf)

In deep learning, understanding exactly what's happening within the layers can be challenging, especially with so many hyperparameters that affect outcomes. Achieving the best performance often comes down to experience and trial and error. In my thesis, I found that reducing the image size to two hundred fifty-six by two hundred fifty-six pixels led to better segmentation results. improved segmentation results but going smaller to one hundred twenty-eight pixels led to poorer results.

