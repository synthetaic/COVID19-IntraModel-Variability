# Abstract

X-ray and computed tomography (CT) scanning technologies for COVID-19 screening have gained significant traction in AI research since the start of the coronavirus pandemic. Despite these continuous advancements for COVID-19 screening, many concerns remain about model reliability when used in a clinical setting. Much has been published, but with limited transparency in expected model performance. We set out to address this limitation through a set of experiments to quantify baseline performance metrics and variability for COVID-19 detection in chest x-ray for 12 common deep learning architectures. Specifically, we adopted an experimental paradigm controlling for train-validation-test split and model architecture where the source of prediction variability originates from model weight initialization, random data augmentation transformations, and batch shuffling. Each model architecture was trained 5 separate times on identical train-validation-test splits of a publicly available x-ray image dataset provided by Cohen et al. (2020). Results indicate that even within model architectures, model behavior varies in a meaningful way between trained models. Best performing models achieve a false negative rate of 3 out of 20 for detecting COVID-19 in a hold-out set. While these results show promise in using AI for COVID-19 screening, they further support the urgent need for diverse medical imaging datasets for model training in a way that yields consistent prediction outcomes. It is our hope that these modeling results accelerate work in building a more robust dataset and a viable screening tool for COVID-19.

\begin{table}[ht]
	\captionsetup{font=small}
	\centering
	\resizebox{\columnwidth}{!}{%
	\begin{tabular}{lrrrrrr}
	 \hline
	Architecture & TPR & FPR & FNR & PPV & F1 & ACC \\ 
	  \hline
	mobilenet\_v2 & 0.75 & 0.022 & 0.25 & 0.334 & 0.455 & 0.869 \\ 
	  densenet121 & 0.71 & \textbf{0.009} & 0.29 & \textbf{0.539} & 0.606 & 0.878 \\ 
	  resnet18 & 0.77 & 0.026 & 0.23 & 0.296 & 0.422 & 0.877 \\ 
	  densenet169 & 0.65 & 0.019 & 0.35 & 0.323 & 0.430 & 0.871 \\ 
	  densenet201 & 0.79 & 0.010 & 0.21 & 0.519 & \textbf{0.625} & \textbf{0.884} \\ 
	  resnext50 & \textbf{0.80} & 0.012 & \textbf{0.20} & 0.481 & 0.596 & 0.868 \\ 
	  resnet50 & 0.71 & 0.017 & 0.29 & 0.384 & 0.489 & 0.871 \\ 
	  resnet101 & 0.73 & 0.021 & 0.27 & 0.335 & 0.456 & 0.879 \\ 
	  resnet152 & 0.63 & 0.014 & 0.37 & 0.391 & 0.481 & 0.878 \\ 
	  wideresnet50 & 0.78 & 0.015 & 0.22 & 0.418 & 0.544 & 0.865 \\ 
	  resnext101 & \textbf{0.80} & 0.019 & \textbf{0.20} & 0.369 & 0.505 & 0.881 \\ 
	  wideresnet101 & 0.78 & 0.021 & 0.22 & 0.336 & 0.468 & 0.874 \\ 
	  All Combined & \textbf{0.80} & \textbf{0.009} & \textbf{0.20} & 0.533 & \textbf{0.640} & \textbf{0.894} \\ 
	   \hline
	\end{tabular}
	}
	\caption{Average performance metrics by model architecture. Note that all metrics except ACC, which is multiclass accuracy, are for COVID-19 detection only (i.e., binary classification). TPR: true positive rate (or recall); FPR: false positive rate; FNR: false negative rate; PPV: positive predictive value (or precision); F1: F1-score; ACC: overall accuracy (TP+TN)/n.}
\label{tab:accmetrics}
\end{table}
