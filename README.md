# LECLIP: Boosting Zero-Shot Anomaly Detection with Local Enhanced CLIP
# abstract:
Zero-Shot Anomaly Detection (ZSAD) is a critical task that detects anomalies without any training samples from the target application, which is crucial for applications in diverse fields such as industrial quality control and medical imaging analysis. Recent advances have seen the application of Contrastive vision-Language Pretraining (CLIP) in ZSAD, exploiting its robust visual-linguistic alignment and zero-shot learning capabilities. However, CLIP is primarily designed for natural
image classification, emphasizing global visual embeddings, while anomaly detection requires more accurate representation of anomalous regions and more precise local visual embeddings. To overcome these limitations, this paper proposes the Local
Enhanced CLIP (LECLIP) framework for ZSAD. LECLIP incorporates a Local Alignment Module that divides images into blocks and aligns them with learnable text embeddings, ensuring precise relevance expression. Furthermore, a trainingfree Echo-Attention is proposed to complement the traditional QKV attention, enabling the model to capture both global and local image details effectively, thus providing a more accurate and detailed image representation. Experimental results show
that LECLIP achieves superior performance on 15 challenging datasets, including 6 industrial datasets and 9 medical datasets.
# Overview of LECLIP
![image](https://github.com/user-attachments/assets/b5604e0c-0e09-4800-b2dd-63543ddd34f2)
# Result
<img width="1169" height="361" alt="image" src="https://github.com/user-attachments/assets/35cbd2a2-73bb-43fc-84a4-7dbd03da03bc" />
<img width="1161" height="312" alt="image" src="https://github.com/user-attachments/assets/98cacbcc-08db-44e7-a4a2-c2acb70b8e11" />

# Requirement
 `pip install -r requirements.txt` 
# Prepare
Prepare the backbone ViT-L-14-336.pt under the AnomalyCLIP_lib path.  
Prepare the JSON file for the dataset. If none can be generated in dataset.py.
# Training（Zero-shot）
We test all datasets by training once on MVTec AD. For MVTec AD, LECLIP is trained on VisA.  
Fast running  
```python 
   python train.py
```
Batch processing  
```python 
  bash train.sh
```
# Test
Fast running  
```python
  python test.py
```
Batch processing
```python
  bash test.sh
```
