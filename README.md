# Camera Eats First: A Food Aesthetics Library 

High Aesthetic Score      |  Low Aesthetic Score
:-------------------------:|:-------------------------:	
![](images/top-images-collage.png)  |  ![](images/bottom-images-collage.png)	


## Introduction

Food aesthetics refers to the sensory gastronomic experience from food presentation such as plating, decorating and styling ([Schifferstein et al., 2020](https://www.tandfonline.com/doi/full/10.1080/15428052.2020.1824833)).

In this paper, by leveraging advanced computer vision and deep learning techniques, we are able to extract meaningful informational content from food images that pertains to the perception of food, which is a long-acknowledged phenomenon as shown in the practices of food photography. 

Instead of relying on a multitude of standard attributes stemming from the photographic literature (e.g., brightness, contrast, rule of thirds, diagonal dominance, etc.), our method leverages a unique one-take-all score, which is more straightforward to understand and more accessible to correlate with other target variables such as restaurants’ page metrics (number of visitors, likes, etc.).

Below, we provide the guidance on how to run the trained model for your set of food images. 

#### Paper: [Camera Eats First: Exploring Food Aesthetics Portrayed on Social Media using Deep Learning](https://www.google.com) [Coming on Early Cite soon]


## Running the Trained Model

### Requirements

1. Install [Python](https://www.python.org/);
2. Git clone this repository to your machine: click the green button "code" in this project's homepage and follow the steps;
3. Install the following dependencies

Dependencies (for Python >=3.10)
```{bash}
tensorflow==2.9
numpy==1.23.0
Pillow==9.2.0
opencv-python==4.6.0.66
```
How to install those:

Once cloned the repository, navigate with the terminal within the project folder homepage and run:
```{bash}
pip3 install -r requirements.txt
```
4. Store the food images in .jpeg format in an image folder in your machine and copy the absolute path. 

An example of an absolute path: /Users/John/Desktop/images

5. Run the model

In the terminal run:
```{bash}
python3 run.py /Users/John/Desktop/images
```

6. The output csv is stored in the output/ folder within this project. Every time the model is called, a new csv file is generated. 

The csv file has two columns: the first one containing the images names, and the second one the aesthetic scores in the 0 (lowest) to 1 (highest) range. 

## Cite Us
## Cite us

```bibtex
@article{gambetti2022,
    title={Camera Eats First: Exploring Food Aesthetics Portrayed on Social Media using Deep Learning},
    author={Alessandro Gambetti and Qiwei Han},
    year={2022},
    journal={International Journal of Contemporary Hospitality Management},
    volume={Ahead of Print},
    number={Ahead of Print},
} 
```

## Next Releases
- pip dedicated package (work in progress)