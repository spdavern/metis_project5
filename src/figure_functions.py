import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

def prcurve(model,y_test,y_pred):
    average_precision=average_precision_score(y_test,y_pred)
    print('Avg Precision: {0:0.4f}'.format(average_precision))
    precision, recall, _ = precision_recall_curve(testy,p_te[:, 1])
    no_skill = len(testy[testy==1]) / len(testy)

    fig = plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b',\
                    label='Random Forest')
    plt.plot([0, 1], [no_skill, no_skill], color='r', linestyle='--', label='No Skill')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend()
    # plt.tight_layout()
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision), fontsize=16);
    fig.savefig('./reports/figures/RandomForest_PrecisionRecallCurve.svg',\
                format='svg', dpi=1200, transparent=True);
    return