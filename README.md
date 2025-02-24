# LECLIP: Boosting Zero-Shot Anomaly Detection with Local Enhanced CLIP
# abstract:
Zero-Shot Anomaly Detection (ZSAD) is a critical task that detects anomalies without any training samples from the target application, which is crucial for applications in diverse fields such as industrial quality control and medical imaging analysis. Recent advances have seen the application of Contrastive vision-Language Pretraining (CLIP) in ZSAD, exploiting its robust visual-linguistic alignment and zero-shot learning capabilities. However, CLIP is primarily designed for natural
image classification, emphasizing global visual embeddings, while anomaly detection requires more accurate representation of anomalous regions and more precise local visual embeddings. To overcome these limitations, this paper proposes the Local
Enhanced CLIP (LECLIP) framework for ZSAD. LECLIP incorporates a Local Alignment Module that divides images into blocks and aligns them with learnable text embeddings, ensuring precise relevance expression. Furthermore, a trainingfree Echo-Attention is proposed to complement the traditional QKV attention, enabling the model to capture both global and local image details effectively, thus providing a more accurate and detailed image representation. Experimental results show
that LECLIP achieves superior performance on 15 challenging datasets, including 6 industrial datasets and 9 medical datasets.
# Overview of LECLIP
![image](https://github.com/user-attachments/assets/26642f00-969c-4e21-8a89-95421b1ac1b8)
# Result
![image](https://github.com/user-attachments/assets/4cf44c39-c4a0-4676-8bf1-7a746332df53)
![image](https://github.com/user-attachments/assets/1c32e1a9-dc06-47ce-a918-10cc4f62953c)

# Training（Zero-shot）
We test all datasets by training once on MVTec AD. For MVTec AD, LECLIP is trained on VisA.
 `python train.py` 
# Test
 `python test.py` 
