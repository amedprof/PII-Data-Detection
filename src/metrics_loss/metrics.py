import pandas as pd
import numpy as np

from sklearn.metrics import fbeta_score

import numpy as np
import pandas as pd
import pandas.api.types

import metrics_loss.kaggle_metric_utilities as ku

import sklearn.metrics

from typing import Sequence, Union, Optional


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float, labels: Optional[Sequence]=None, pos_label: Union[str, int]=1, average: str='micro', weights_column_name: Optional[str]=None) -> float:
    '''
    Wrapper for https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
    Compute the F-beta score.

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    The `beta` parameter determines the weight of recall in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (``beta -> 0`` considers only precision, ``beta -> +inf``
    only recall).

    Parameters
    ----------
    solution : 1d DataFrame, or label indicator array / sparse matrix
    Ground truth (correct) target values.

    submission : 1d DataFrame, or label indicator array / sparse matrix
    Estimated targets as returned by a classifier.

    beta : float
    Determines the weight of recall in the combined score.

    labels : optional array-like, default=None
    The set of labels to include when ``average != 'binary'``, and their
    order if ``average is None``. Labels present in the data can be
    excluded, for example to calculate a multiclass average ignoring a
    majority negative class, while labels not present in the data will
    result in 0 components in a macro average. For multilabel targets,
    labels are column indices. By default, all labels in ``y_true`` and
    ``y_pred`` are used in sorted order.

    pos_label : str or int, default=1
    The class to report if ``average='binary'`` and the data is binary.
    If the data are multiclass or multilabel, this will be ignored;
    setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
    scores for that label only.

    average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None,             default='binary'
    This parameter is required for multiclass/multilabel targets.
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'binary'``:
    Only report results for the class specified by ``pos_label``.
    This is applicable only if targets (``y_{true,pred}``) are binary.
    ``'micro'``:
    Calculate metrics globally by counting the total true positives,
    false negatives and false positives.
    ``'macro'``:
    Calculate metrics for each label, and find their unweighted
    mean.  This does not take label imbalance into account.
    ``'weighted'``:
    Calculate metrics for each label, and find their average weighted
    by support (the number of true instances for each label). This
    alters 'macro' to account for label imbalance; it can result in an
    F-score that is not between precision and recall.
    ``'samples'``:
    Calculate metrics for each instance, and find their average (only
    meaningful for multilabel classification where this differs from
    `accuracy_score`).

    weights_column_name: optional str, the name of the sample weights column in the solution file.

    <https://en.wikipedia.org/wiki/F1_score>`_.

    Examples
    --------

    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name, average='macro', beta=0.5)
    0.23...
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name, average='micro', beta=0.5)
    0.33...
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name, average='weighted', beta=0.5)
    0.23...
    '''
    # Skip sorting and equality checks for the row_id_column since that should already be handled
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    sample_weight = None
    if weights_column_name:
        if weights_column_name not in solution.columns:
            raise ValueError(f'The solution weights column {weights_column_name} is not found')
        sample_weight = solution.pop(weights_column_name).values
        if not pandas.api.types.is_numeric_dtype(sample_weight):
            raise ParticipantVisibleError('The solution weights are not numeric')

    if not((len(submission.columns) == 1) or (len(submission.columns) == len(solution.columns))):
        raise ParticipantVisibleError(f'Invalid number of submission columns. Found {len(submission.columns)}')

    solution = solution.values
    submission = submission.values

    score_result = ku.safe_call_score(sklearn.metrics.fbeta_score, solution, submission, beta=beta, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)

    return score_result


LABEL2TYPE = ('NAME_STUDENT','EMAIL','USERNAME','ID_NUM', 'PHONE_NUM','URL_PERSONAL','STREET_ADDRESS','O')
LABEL = {l: t for l, t in enumerate(LABEL2TYPE)}

def pii_fbeta_score_v2(pred_df, gt_df, beta=5):
    """
    Parameters:
    - pred_df (DataFrame): DataFrame containing predicted PII labels.
    - gt_df (DataFrame): DataFrame containing ground truth PII labels.
    - beta (float): The beta parameter for the F-beta score, controlling the trade-off between precision and recall.

    Returns:
    - float: Micro F-beta score.
    """   

    df = pred_df.merge(gt_df, how="outer", on=["document", "token"], suffixes=("_p", "_g"))
    df["cm"] = ""
    
    df.loc[df.label_gt.isna(), "cm"] = "FP"
    df.loc[df.label_pred.isna(), "cm"] = "FN"

    # df.loc[(df.label_gt.notna()) & (df.label_gt != df.label_pred), "cm"] = "FNFP"
    df.loc[(df.label_gt.notna() & df.label_pred.notna()) & (df.label_gt != df.label_pred), "cm"] = "FNFP" # CHANGED
    
    df.loc[
        (df.label_pred.notna()) & (df.label_gt.notna()) & (df.label_gt == df.label_pred), "cm"
    ] = "TP"

    FP = (df["cm"].isin({"FP", "FNFP"})).sum()
    FN = (df["cm"].isin({"FN", "FNFP"})).sum()
    TP = (df["cm"] == "TP").sum()
    s_micro = (1+(beta**2))*TP/(((1+(beta**2))*TP) + ((beta**2)*FN) + FP)

    return s_micro



LABEL2TYPE = ('NAME_STUDENT','EMAIL','USERNAME','ID_NUM', 'PHONE_NUM','URL_PERSONAL','STREET_ADDRESS','O')
LABEL = {l: t for l, t in enumerate(LABEL2TYPE)}

def score_feedback(pred_df, gt_df):
    df = pred_df.merge(gt_df,how='outer',on=['document',"token"],suffixes=('_p','_g'))

    df['status'] = "TN"

    df.loc[df.label_gt.isna(),'status'] = "FP"
    df.loc[df.label_pred.isna(),'status'] = "FN"
    df.loc[(df.label_gt.notna()) & (df.label_gt!=df.label_pred),'status'] = "FN"
    df.loc[(df.label_pred.notna()) & (df.label_gt.notna()) & (df.label_gt==df.label_pred),'status'] = "TP"

    FP = (df['status'].isin(["FP"])).sum()
    FN = (df['status'].isin(["FN"])).sum()
    TP = (df['status']=="TP").sum()

    s_micro = (1+(5**2))*TP/(((1+(5**2))*TP) + ((5**2)*FN) + FP)

    df["cm"] = ""
    
    df.loc[df.label_gt.isna(), "cm"] = "FP"
    df.loc[df.label_pred.isna(), "cm"] = "FN"

    # df.loc[(df.label_gt.notna()) & (df.label_gt != df.label_pred), "cm"] = "FNFP"
    df.loc[(df.label_gt.notna() & df.label_pred.notna()) & (df.label_gt != df.label_pred), "cm"] = "FNFP" # CHANGED
    
    df.loc[
        (df.label_pred.notna()) & (df.label_gt.notna()) & (df.label_gt == df.label_pred), "cm"
    ] = "TP"

    FP = (df["cm"].isin({"FP", "FNFP"})).sum()
    FN = (df["cm"].isin({"FN", "FNFP"})).sum()
    TP = (df["cm"] == "TP").sum()
    s_micro_new = (1+(5**2))*TP/(((1+(5**2))*TP) + ((5**2)*FN) + FP)


    dic_class = {}
    classes = gt_df['label'].unique()
    for c in classes:
        
        dx = pred_df[pred_df.label==c].merge(gt_df[gt_df.label==c],how='outer',on=['document',"token"],suffixes=('_p','_g'))
#         dx = df[(df.label_gt.isna()) | (df.label_g==c) | (df.label_p==c)].reset_index()
        dx["cm1"] = ""
    
        dx.loc[dx.label_gt.isna(), "cm1"] = "FP"
        dx.loc[dx.label_pred.isna(), "cm1"] = "FN"

        # df.loc[(df.label_gt.notna()) & (df.label_gt != df.label_pred), "cm"] = "FNFP"
        dx.loc[(dx.label_gt.notna() & dx.label_pred.notna()) & (dx.label_gt != dx.label_pred), "cm1"] = "FNFP" # CHANGED

        dx.loc[
            (dx.label_pred.notna()) & (dx.label_gt.notna()) & (dx.label_gt == dx.label_pred), "cm1"
        ] = "TP"

        FP = (dx["cm1"].isin({"FP", "FNFP"})).sum()
        FN = (dx["cm1"].isin({"FN", "FNFP"})).sum()
        TP = (dx["cm1"] == "TP").sum()
        s = (1+(5**2))*TP/(((1+(5**2))*TP) + ((5**2)*FN) + FP)
    
#         s = (1+(5**2))*tp/(((1+(5**2))*tp) + ((5**2)*fn) + fp) if tp+fp+fn !=0 else -1
        dic_class[LABEL[c]] = s
    # df.loc[(df.label_pred.notna()) & (df.label_gt.notna()) & (df.label_gt==df.label_pred),'status'] = "TP"


    # s_micro = fbeta_score(df['label'].values, df['label_pred'].values, average='micro', beta=5)
    # s_macro = fbeta_score(df['label'].values, df['label_pred'].values, average='macro', beta=5)
    return s_micro_new,s_micro,dic_class




from collections import defaultdict
from typing import Dict


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


def compute_metrics(pred_df, gt_df):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """
    
    references = {(row.document, row.token, row.label) for row in gt_df.itertuples()}
    predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

    score_per_type = defaultdict(PRFScore)
    references = set(references)

    for ex in predictions:
        pred_type = ex[-1] # (document, token, label)
        if pred_type != 'O':
            pred_type = pred_type[2:] # avoid B- and I- prefix
            
        if pred_type not in score_per_type:
            score_per_type[pred_type] = PRFScore()

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1

    for doc, tok, ref_type in references:
        if ref_type != 'O':
            ref_type = ref_type[2:] # avoid B- and I- prefix
        
        if ref_type not in score_per_type:
            score_per_type[ref_type] = PRFScore()
        score_per_type[ref_type].fn += 1

    totals = PRFScore()
    
    for prf in score_per_type.values():
        totals += prf

    return {
        "f5_prec": totals.precision,
        "f5_rec": totals.recall,
        "f5_micro": totals.f5,
        "ents_per_type": {k: v.to_dict()['f5'] for k, v in score_per_type.items() if k!= 'O' },
    }