<h1 align="center">M.A.R.I.S</h1>
<h2 align="center"> An Marine Automated Recognition and Identification System based on Artificial Intelligence Methods</h2>

## Abstract
he necessity of employing autonomous software and algorithms for ecosystem observation has significantly increased with the advancement of neural pattern recognition and machine learning technologies. 
The Marine Automated Recognition and Identification System (MARIS) is designed to address this need by providing a comprehensive and optimized platform for the detailed study and monitoring of marine flora and fauna. Leveraging state-of-the-art neural networks and computer vision techniques, MARIS aims to deliver accurate and real-time identification of marine species, track biodiversity, and assess the health of marine ecosystems. Beyond ecological monitoring, MARIS also incorporates functionalities for detecting and classifying marine debris, such as plastics and other pollutants, thereby contributing to efforts in pollution control and environmental protection. 
the system’s adaptability allows it to be utilized in various marine research applications, including habitat mapping, species behavior analysis, and the assessment of human impact on marine environments. By integrating these advanced features, MARIS strives to enhance our understanding of marine ecosystems and support conservation initiatives through innovative, autonomous technological solutions.

## How to Use
  For start the process of MARIS:
  ```bash
  python3 MARIS.py --image IMAGE_PATH
  ```
  For start the mixed-version of MARIS:
  ```bash
  python3 MARIS-FT_MIX.py --image IMAGE_PATH --object object (to detect & segment)
  ```
  Access Paligemma-3b-mix model:
  Link: https://huggingface.co/google/paligemma-3b-mix-448

## Fine-Tune The Model
For Fine-Tuning the model you can use the fine-tuning.py file for training from a coral dataset.
The dataset have to beein in paligemma format and modify the value inside the script.
 ```bash
  python3 fine-tune.py
  ```
The dataset used for our cases: https://universe.roboflow.com/merg/coral-bamqm/dataset/2
For transform the fine-tuned model into the huggin-face format use the script from this repo.
Link : https://github.com/huggingface/transformers/blob/main/src/transformers/models/paligemma/convert_paligemma_weights_to_hf.py