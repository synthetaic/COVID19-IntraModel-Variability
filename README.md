# Abstract

X-ray and computed tomography (CT) scanning technologies for COVID-19 screening have gained significant traction in AI research since the start of the coronavirus pandemic. Despite these continuous advancements for COVID-19 screening, many concerns remain about model reliability when used in a clinical setting. Much has been published, but with limited transparency in expected model performance. We set out to address this limitation through a set of experiments to quantify baseline performance metrics and variability for COVID-19 detection in chest x-ray for 12 common deep learning architectures. Specifically, we adopted an experimental paradigm controlling for train-validation-test split and model architecture where the source of prediction variability originates from model weight initialization, random data augmentation transformations, and batch shuffling. Each model architecture was trained 5 separate times on identical train-validation-test splits of a publicly available x-ray image dataset provided by Cohen et al. (2020). Results indicate that even within model architectures, model behavior varies in a meaningful way between trained models. Best performing models achieve a false negative rate of 3 out of 20 for detecting COVID-19 in a hold-out set. While these results show promise in using AI for COVID-19 screening, they further support the urgent need for diverse medical imaging datasets for model training in a way that yields consistent prediction outcomes. It is our hope that these modeling results accelerate work in building a more robust dataset and a viable screening tool for COVID-19.

<table>
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
