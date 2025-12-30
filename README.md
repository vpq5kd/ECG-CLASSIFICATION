# Classification of ECG Diagnoses with a Convulotional Neural Network.
<img width="1500" height="500" alt="image" src="https://github.com/user-attachments/assets/d70b688a-87ac-4b87-8b2f-9d54fdb0bc31" />
<figcaption>A confusion matrix for the 5 different 'diagnostic superclasses' outlined by PTB-XL</figcaption>

## Problem

Automated ECG classification is a widely used tool in a variety of clinical environments, including the pre-hospital environment, where often a lack of providers certified to interpret ECGs means a reliance on a computer to determine whether or not an ECG is alarming. Though there have been a variety of improvements in this space, these algorithms rely primarily on discrete diagnositic criteria and often make mistakes. Neural Network based classification, using large cardiologist validated ECG diagnoses, takes a different approach to this problem of classification. Instead of asking for the precise amount of milimeters the 'j-point' of the ST-Segment reaches, or whether it is concave up or concave down, it takes the ECG and sees whether it's total data fits in with other ECGs that were interpreted by a human being. By applying this neural network architecture to this problem of classification you are taking the brain of a distinguished physician and placing it in the hands of clinicians without the training to interpret ECGs. By doing so we may be able to achieve higher accuracy in computational ECG classification and ultimately better outcomes for patients. 

-----

## CNN Design and Dataset Choice

### CNN Design

CNN design is a highly complicated process that relies on large amounts of _a posteriori_ knowledge to create effective results. For the first iteration of CNN design for this project, which is the one currently displayed, I decided to go with architecture specifically built for the PTB-XL dataset outlined by three researches from the UTP University of Science and Technology in Poland [^1]. These researchers designed a 5 layer convulotional neural network with one flattening layer and one fully-connected layer, and achieved similar results to my implementation of their research. Given that this is a relatively simplisitic CNN design, it's accuracy provides promising insight into the utility of CNN classification as methodology for clinical use. 

### Dataset Choice

Since my original imeptus for this project was to use human cardiologist validated ECGs for clinical ECG classification, I chose the PTB-XL dataset as it provided over 20000 ECGs with diagnositics validated by up to two cardiologists [^2][^3]. This data was then split in the method that is outlined in the the `example_physionet.py` file in the Physionet version of the PTB-XL database, which, per Physionet, was the method in which the researchers intended for their data to be split.

-----

## Making, Using, and Replicating my Code

This code is split into the three files `*_training.py`, `*_testing.py`, and `*_results.py` with some skeleton code taken from various resources[^4][^5]. To build the neural network, ensuring of course that you have the appropriate dependecies installed along with the dataset downloaded, run the `*_training.py` file. This will save the neural network which can later be used by `*_results.py` (and by extension `*_testing.py`) when you run `*_results.py` to see the results of the CNN. `*_testing.py` is a seperate file intended to make the code cleaner and does not need to be run independently to create or use the neural network. 

-----

## Results

Using the CNN from the researches at UTP University, I achieved a global accuracy of around 75%, far too low to consider using this technique at scale or using it to replace discrete ECG classification in clinical practice. That being said, this metric does show _promising_ results for this technique in general, and perhaps with further design iterations a higher global accuracy could be achieved. Further, in the pre-hospital enviornment, which is the target envionrment for a later end product (as defined in the next section) ECG classification only needs to determine whether or not an ECG indicates myocordial infarcation. This need points to a possible reduction of the data to a binary model of whether or not an ECG is an MI, which reduces the complexity of the CNN and could give more accuracate results.

-----

## End-Product

Pre-hospital ECG acquisition capabale monitors usually automatically interpet the results with discrete criteria, as mentioned earlier. I see this technique as a potentional addition or even replacement for the back-end engine that powers these interpreters, ultimately decreasing 'door-to-balloon' times of patients with acute myocardial ischemia. 

-----

**Disclaimer:** ChatGPT was used in the development of this program in a similar manner to the traditional Google search or Stack Overflow query. That is, of course, for edits and assistance with syntax, coding convetions, python standard functions, etc. Except where otherwise noted, it was not used for the development of original thoughts or algorithmic code. All of my code can be explained thoroughly by me if necessary. 

[^1]: Śmigiel, S., Pałczyński, K., & Ledziński, D. (2021). ECG Signal Classification Using Deep Learning Techniques Based on the PTB-XL Dataset. Entropy, 23(9), 1121. https://doi.org/10.3390/e23091121
[^2]: Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kfzx-aw45
[^3]: Wagner, P., Strodthoff, N., Bousseljot, RD. et al. PTB-XL, a large publicly available electrocardiography dataset. Sci Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6
[^4]: Kuriakose, J. (2022, January 31). A simple neural network classifier using pytorch, from scratch | by Jeril Kuriakose | Analytics Vidhya | Medium. Medium. https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2 
[^5]: Luna, J. (2025, February 27). Pytorch CNN tutorial: Build & Train Convolutional Neural Networks in python | datacamp. datacamp. https://www.datacamp.com/tutorial/pytorch-cnn-tutorial 

