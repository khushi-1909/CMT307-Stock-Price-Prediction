import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, precision_score, recall_score, make_scorer, accuracy_score, f1_score
import matplotlib.pyplot as plt
def present_model_results(y_test, y_pred_set):
  thresh=0.51
  y_pred_class_set = []
  roc_auc_set = []
  cmats = []
  model_colours = ['#2ea647', '#472ea6', '#a6472e']
  models = ['RF', 'CNN', 'Hybrid']
  fig, ax = plt.subplots(2,3, figsize = (10,6), layout = 'constrained')
  ax = ax.flatten()
  ax[0].set_ylabel('True Positive Rate')
  ax[0].set_xlabel('False Positive Rate')
  ax[0].set_title('ROC Curve')
  ax[2].set_ylabel('Precision')
  ax[2].set_xlabel('Recall')
  ax[2].set_title('Precision-Recall')
  ax[1].set_ylabel('AUC')
  ax[1].set_title('Area Under ROC Curve (AUC)')
  ax[1].set_xlabel('Model')
  ax[1].set_xticks([0,1,2])
  ax[1].set_xticklabels(['I: RF', 'II: CNN', 'III: Hybrid'])
  #random chance line
  ax[0].plot(np.linspace(0,1,100), np.linspace(0,1,100), label = 'AUC=0.5', c = 'k', ls = 'dashed')
  
  for i, model_pred in enumerate(y_pred_set):
     y_pred_class_set.append((np.where(model_pred > thresh, 1,0)))
     y_test_ = y_test[i]
     fpr, tpr, _ = roc_curve(y_test_, model_pred)
     #recall = recall_score(y_test,y_pred_class_set[i])
     #precision = precision_score(y_test,y_pred_class_set[i])
     rec, prec, _ = precision_recall_curve(y_test_, model_pred, drop_intermediate=False)
     roc_auc = auc(fpr,tpr)
     roc_auc_set.append(roc_auc)
     ax[2].plot(rec, prec, c = model_colours[i], label = models[i]) #precision recall curve
     ax[0].plot(fpr, tpr, c = model_colours[i], label = models[i]) # ROC curve
     cmats.append(confusion_matrix(y_test_,y_pred_class_set[i], normalize = 'true'))
  # auc bar chart
  b = ax[1].bar([0,1,2], roc_auc_set, color = model_colours)
  ax[1].bar_label(b, label_type='center', fmt = '%.3f',color = 'w' )
  ax[2].legend(loc='center left')
  ax[0].legend()
  for j,a in enumerate(ax[3:]):
    im = a.imshow(cmats[j], cmap = 'cividis', vmin=0, vmax = 1)
    a.set_yticks([0,1])
    a.set_xlabel('Ground truth')
    a.set_ylabel('Prediction')
    a.set_xticks([0,1])
    a.set_title(f'Confusion matrix, {models[j]}')
    a.text(-0.2,-0.2, round(cmats[j][0][0],3), color = 'w')
    a.text(-0.2, 0.8, round(cmats[j][1][0],3), color = 'w')
    a.text(0.8, -0.2,round(cmats[j][0][1],3), color = 'w')
    a.text(0.8, 0.8,round(cmats[j][1][1],3), color = 'w')
  fig.colorbar(im, label = 'Fraction')
  fig.savefig('Results_Figure.png')
  return 0