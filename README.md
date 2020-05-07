# Abstract

X-ray and computed tomography (CT) scanning technologies for COVID-19 screening have gained significant traction in AI research since the start of the coronavirus pandemic. Despite these continuous advancements for COVID-19 screening, many concerns remain about model reliability when used in a clinical setting. Much has been published, but with limited transparency in expected model performance. We set out to address this limitation through a set of experiments to quantify baseline performance metrics and variability for COVID-19 detection in chest x-ray for 12 common deep learning architectures. Specifically, we adopted an experimental paradigm controlling for train-validation-test split and model architecture where the source of prediction variability originates from model weight initialization, random data augmentation transformations, and batch shuffling. Each model architecture was trained 5 separate times on identical train-validation-test splits of a publicly available x-ray image <a href="https://github.com/ieee8023/covid-chestxray-dataset">dataset</a> provided by Cohen et al. (2020). Results indicate that even within model architectures, model behavior varies in a meaningful way between trained models. Best performing models achieve a false negative rate of 3 out of 20 for detecting COVID-19 in a hold-out set. While these results show promise in using AI for COVID-19 screening, they further support the urgent need for diverse medical imaging datasets for model training in a way that yields consistent prediction outcomes. It is our hope that these modeling results accelerate work in building a more robust dataset and a viable screening tool for COVID-19.

## Citation

The work described here has been __<a href="https://arxiv.org/abs/2005.02167">published on arXiv</a>__.

Cite our work (BibTeX):
```
@misc{goodwin2020intramodel,
    title={Intra-model Variability in COVID-19 Classification Using Chest X-ray Images},
    author={Brian D Goodwin and Corey Jaskolski and Can Zhong and Herick Asmani},
    year={2020},
    eprint={2005.02167},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

Performance metrics for all trained models (on average) shown in the table below. These _model weights_ and _training/testing data_ from the published study can be accessed and downloaded via 

__<a href="https://covidresearch.ai/datasets/dataset?id=2">covidresearch.ai</a>__

and code is available via

__<a href="https://github.com/synthetaic/COVID19-IntraModel-Variability">github.com/synthetaic</a>__

# Using the Notebooks

These notebooks are designed to walk you through training networks and/or loading in pre-trained model weights from the study. Download the dataset used for scoring at <a href="https://covidresearch.ai/datasets/dataset?id=2#2Files">covidresearch.ai</a>, which contains the train-validation-test split with 20 COVID-19 positive patient images held out in the test set. We've provided comments in each of the notebooks to help explain the process for each code chunk.

# Results

Refer to our publication for results and details regarding the variability between networks and other performance metrics. Otherwise, the table below highlights key performance metrics on existing x-ray dataset for detecting COVID-19. However, since the publication of this work, the __<a href="https://github.com/ieee8023/covid-chestxray-dataset">dataset</a>__ sample size has increased.

<table>
<caption>Table 1 Average performance metrics by model architecture. Note that all metrics except ACC, which is multiclass accuracy, are for COVID-19 detection only (i.e., binary classification). TPR: true positive rate (or recall); FPR: false positive rate; FNR: false negative rate; PPV: positive predictive value (or precision); F1: F1-score; ACC: overall accuracy (TP+TN)/n.</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> Arch </th>
   <th style="text-align:right;"> TPR </th>
   <th style="text-align:right;"> FPR </th>
   <th style="text-align:right;"> FNR </th>
   <th style="text-align:right;"> PPV </th>
   <th style="text-align:right;"> F1 </th>
   <th style="text-align:right;"> ACC </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> mobilenet_v2 </td>
   <td style="text-align:right;"> 0.75 </td>
   <td style="text-align:right;"> 0.0222968 </td>
   <td style="text-align:right;"> 0.25 </td>
   <td style="text-align:right;"> 0.3344489 </td>
   <td style="text-align:right;"> 0.4548416 </td>
   <td style="text-align:right;"> 0.8694500 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> densenet121 </td>
   <td style="text-align:right;"> 0.71 </td>
   <td style="text-align:right;"> 0.0087307 </td>
   <td style="text-align:right;"> 0.29 </td>
   <td style="text-align:right;"> 0.5394747 </td>
   <td style="text-align:right;"> 0.6063897 </td>
   <td style="text-align:right;"> 0.8775348 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resnet18 </td>
   <td style="text-align:right;"> 0.77 </td>
   <td style="text-align:right;"> 0.0260578 </td>
   <td style="text-align:right;"> 0.23 </td>
   <td style="text-align:right;"> 0.2960767 </td>
   <td style="text-align:right;"> 0.4219726 </td>
   <td style="text-align:right;"> 0.8774023 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> densenet169 </td>
   <td style="text-align:right;"> 0.65 </td>
   <td style="text-align:right;"> 0.0186702 </td>
   <td style="text-align:right;"> 0.35 </td>
   <td style="text-align:right;"> 0.3230125 </td>
   <td style="text-align:right;"> 0.4301389 </td>
   <td style="text-align:right;"> 0.8709079 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> densenet201 </td>
   <td style="text-align:right;"> 0.79 </td>
   <td style="text-align:right;"> 0.0099396 </td>
   <td style="text-align:right;"> 0.21 </td>
   <td style="text-align:right;"> 0.5193145 </td>
   <td style="text-align:right;"> 0.6248680 </td>
   <td style="text-align:right;"> 0.8837641 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resnext50 </td>
   <td style="text-align:right;"> 0.80 </td>
   <td style="text-align:right;"> 0.0123573 </td>
   <td style="text-align:right;"> 0.20 </td>
   <td style="text-align:right;"> 0.4809069 </td>
   <td style="text-align:right;"> 0.5960700 </td>
   <td style="text-align:right;"> 0.8683897 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resnet50 </td>
   <td style="text-align:right;"> 0.71 </td>
   <td style="text-align:right;"> 0.0171927 </td>
   <td style="text-align:right;"> 0.29 </td>
   <td style="text-align:right;"> 0.3840065 </td>
   <td style="text-align:right;"> 0.4893698 </td>
   <td style="text-align:right;"> 0.8713055 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resnet101 </td>
   <td style="text-align:right;"> 0.73 </td>
   <td style="text-align:right;"> 0.0205507 </td>
   <td style="text-align:right;"> 0.27 </td>
   <td style="text-align:right;"> 0.3348723 </td>
   <td style="text-align:right;"> 0.4556031 </td>
   <td style="text-align:right;"> 0.8791252 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resnet152 </td>
   <td style="text-align:right;"> 0.63 </td>
   <td style="text-align:right;"> 0.0135662 </td>
   <td style="text-align:right;"> 0.37 </td>
   <td style="text-align:right;"> 0.3911076 </td>
   <td style="text-align:right;"> 0.4805679 </td>
   <td style="text-align:right;"> 0.8780649 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> wideresnet50 </td>
   <td style="text-align:right;"> 0.78 </td>
   <td style="text-align:right;"> 0.0146407 </td>
   <td style="text-align:right;"> 0.22 </td>
   <td style="text-align:right;"> 0.4182053 </td>
   <td style="text-align:right;"> 0.5439933 </td>
   <td style="text-align:right;"> 0.8652087 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resnext101 </td>
   <td style="text-align:right;"> 0.80 </td>
   <td style="text-align:right;"> 0.0185359 </td>
   <td style="text-align:right;"> 0.20 </td>
   <td style="text-align:right;"> 0.3694735 </td>
   <td style="text-align:right;"> 0.5045063 </td>
   <td style="text-align:right;"> 0.8812459 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> wideresnet101 </td>
   <td style="text-align:right;"> 0.78 </td>
   <td style="text-align:right;"> 0.0212223 </td>
   <td style="text-align:right;"> 0.22 </td>
   <td style="text-align:right;"> 0.3360063 </td>
   <td style="text-align:right;"> 0.4678163 </td>
   <td style="text-align:right;"> 0.8735586 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> All Combined </td>
   <td style="text-align:right;"> 0.80 </td>
   <td style="text-align:right;"> 0.0094023 </td>
   <td style="text-align:right;"> 0.20 </td>
   <td style="text-align:right;"> 0.5333333 </td>
   <td style="text-align:right;"> 0.6400000 </td>
   <td style="text-align:right;"> 0.8939695 </td>
  </tr>
</tbody>
</table>
